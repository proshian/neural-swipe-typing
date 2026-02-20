import sys; import os; sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Manager, get_context

import torch
from tqdm import tqdm
from hydra.utils import get_original_cwd
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import SwipeDataset, SwipeDatasetSubset
from feature_extraction.swipe_feature_extractor_factory import swipe_feature_extractor_factory
from logit_processors import VocabularyLogitProcessor
from model import get_model_from_configs
from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizer
from train import validate_d_model
from word_generators import generator_factory


# Keys from train config that are needed for prediction
TRAIN_KEYS_FOR_PREDICT  = frozenset([
    # Model architecture
    'encoder',
    'decoder',
    'swipe_point_embedder',
    'feature_extractor',
    'd_model',
    'num_classes',
    'max_out_seq_len',

    'grid_name',

    # Data paths (for tokenizers, grids, normalization stats)
    'grids_path',
    'keyboard_tokenizer_path',
    'vocab_path',
    'trajectory_features_statistics_path',
    'bounding_boxes_path',

    'experiment_name',
])


def merge_train_and_predict_configs(train_cfg: DictConfig, predict_cfg: DictConfig) -> DictConfig:
    merged = OmegaConf.create({})

    for key in TRAIN_KEYS_FOR_PREDICT:
        if key in train_cfg:
            merged[key] = train_cfg[key]

    # Apply prediction config overrides (non-null values only)
    for key in predict_cfg.keys():
        if predict_cfg[key] is not None:
            merged[key] = predict_cfg[key]

    return merged


@dataclass
class PredictionResult:
    predictions: List[List[Tuple[float, str]]]
    config: dict


def get_vocab(vocab_path: str) -> List[str]:
    with open(vocab_path, 'r', encoding="utf-8") as f:
        return f.read().splitlines()


def read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# --- Multiprocessing Globals & Helpers ---
# It's faster to use global objects than to use partial functions
_worker_word_generator = None
_worker_dataset_subset = None

def worker_init(word_generator, dataset_subset):
    """Initializes each worker process with the required objects."""
    global _worker_word_generator, _worker_dataset_subset
    _worker_word_generator = word_generator
    _worker_dataset_subset = dataset_subset

def worker_predict(i: int) -> List[Tuple[float, str]]:
    """The prediction function executed by each worker."""
    ((swipe_features, _), _) = _worker_dataset_subset[i]
    return _worker_word_generator(swipe_features)
# ---


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    original_cwd = get_original_cwd()

    def resolve_path(path: str) -> str:
        if Path(path).is_absolute():
            return path
        return str(Path(original_cwd) / path)

    # If train_config_path is provided, merge with train config
    train_config_path = cfg.get('train_config_path')
    if train_config_path is not None:
        train_cfg_path = resolve_path(train_config_path)
        train_cfg = OmegaConf.load(train_cfg_path)
        cfg = merge_train_and_predict_configs(train_cfg, cfg)
        print(f"Merged predict config with train config: {train_cfg_path}")

    # Auto-derive output path if not provided
    if cfg.get('output_path') is None:
        exp_name = cfg.experiment_name
        data_name = Path(cfg.data_path).stem
        ckpt_name = Path(cfg.model_weights_path).stem
        cfg.output_path = f"results/predictions/{exp_name}/{data_name}/{cfg.grid_name}/{ckpt_name}.pkl"

    num_workers = cfg.get("num_workers", 1)
    manager = None if num_workers <= 1 else Manager()

    device = torch.device(cfg.device)

    subword_tokenizer = CharLevelTokenizerv2(resolve_path(cfg.vocab_path))
    keyboard_tokenizer = KeyboardTokenizer(resolve_path(cfg.keyboard_tokenizer_path))

    grids = read_json(resolve_path(cfg.grids_path))
    grid = grids[cfg.grid_name]

    feature_extractor_config = OmegaConf.to_container(cfg.feature_extractor, resolve=True)
    trajectory_stats = read_json(resolve_path(cfg.trajectory_features_statistics_path))
    bounding_boxes = read_json(resolve_path(cfg.bounding_boxes_path))

    feature_extractor = swipe_feature_extractor_factory(
        grid=grid,
        keyboard_tokenizer=keyboard_tokenizer,
        trajectory_features_statistics=trajectory_stats,
        bounding_boxes=bounding_boxes,
        grid_name=cfg.grid_name,
        component_configs=feature_extractor_config
    )

    dataset = SwipeDataset(
        data_path=resolve_path(cfg.data_path),
        word_tokenizer=subword_tokenizer,
        grid_name_to_swipe_feature_extractor={cfg.grid_name: feature_extractor},
        use_serialized_list=False
    )
    dataset_subset = SwipeDatasetSubset(dataset, cfg.grid_name)

    # Load all component configs
    input_embedding_config = OmegaConf.to_container(cfg.swipe_point_embedder, resolve=True)
    encoder_config = OmegaConf.to_container(cfg.encoder, resolve=True)
    decoder_config = OmegaConf.to_container(cfg.decoder, resolve=True)

    # Validate d_model
    d_model = cfg.get('d_model')
    if d_model is None:
        raise ValueError(
            "d_model must be specified in prediction config. "
            "It should match the output dimension of the swipe point embedder."
        )
    validate_d_model(d_model, feature_extractor, input_embedding_config)

    model = get_model_from_configs(
        input_embedding_config=input_embedding_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_classes=cfg.num_classes,
        n_word_tokens=len(subword_tokenizer.char_to_idx),
        max_out_seq_len=cfg.max_out_seq_len,
        d_model=d_model,
        device=device,
        weights_path=resolve_path(cfg.model_weights_path)
    )
    model.eval()

    if num_workers > 1:
        model.share_memory()
        dataset_subset.dataset.data_list = manager.list(dataset_subset.dataset.data_list)
        dataset_subset.grid_idxs = manager.list(dataset_subset.grid_idxs)

    logit_processor = None
    if cfg.get('use_vocab_for_generation', False):
        vocab = get_vocab(resolve_path(cfg.vocab_path))
        logit_processor = VocabularyLogitProcessor(
            tokenizer=subword_tokenizer,
            vocab=vocab,
            max_token_id=cfg.num_classes - 1
        )
        if num_workers > 1:
            logit_processor.prefix_to_allowed_ids = manager.dict(logit_processor.prefix_to_allowed_ids)

    generator_params = OmegaConf.to_container(cfg.generator.params, resolve=True)
    word_generator = generator_factory(
        generator_name=cfg.generator.name,
        model=model,
        subword_tokenizer=subword_tokenizer,
        device=device,
        logit_processor=logit_processor,
        generator_params=generator_params
    )

    all_predictions = []
    if num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=worker_init, initargs=(word_generator, dataset_subset)) as pool:
            results_iterator = pool.imap(worker_predict, range(len(dataset_subset)))
            for preds in tqdm(results_iterator, total=len(dataset_subset), desc="Generating predictions (multi-process)"):
                all_predictions.append(preds)
    else:
        for i in tqdm(range(len(dataset_subset)), desc="Generating predictions (single-process)"):
            ((swipe_features, _), _) = dataset_subset[i]
            predictions = word_generator(swipe_features)
            all_predictions.append(predictions)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    result = PredictionResult(predictions=all_predictions, config=config_dict)
    os.makedirs(os.path.dirname(resolve_path(cfg.output_path)), exist_ok=True)
    with open(resolve_path(cfg.output_path), 'wb') as f:
        pickle.dump(result, f)

    print(f"Predictions saved to {cfg.output_path}")


if __name__ == "__main__":
    main()
