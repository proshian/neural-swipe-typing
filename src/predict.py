import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple
from multiprocessing import Pool, Manager, get_context

import torch
from tqdm import tqdm

from dataset import SwipeDataset, SwipeDatasetSubset
from feature_extraction.swipe_feature_extractor_factory import swipe_feature_extractor_factory
from logit_processors import VocabularyLogitProcessor
from model import get_model_from_configs
from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizer
from word_generators import generator_factory


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
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions for swipe typing model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the prediction config file.')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of worker processes for prediction.')
    return parser.parse_args()

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

def predict(config: dict, num_workers: int):
    manager = None if num_workers <= 1 else Manager()

    device = torch.device(config['device'])

    subword_tokenizer = CharLevelTokenizerv2(config['voc_path'])
    keyboard_tokenizer = KeyboardTokenizer(config['keyboard_tokenizer_path'])

    grid = read_json(config['grids_path'])[config['grid_name']]
    swipe_feature_extractor_config = read_json(config['swipe_feature_extractor_factory_config_path'])
    trajectory_stats = read_json(config['trajectory_features_statistics_path'])
    bounding_boxes = read_json(config['bounding_boxes_path'])

    feature_extractor = swipe_feature_extractor_factory(
        grid=grid,
        keyboard_tokenizer=keyboard_tokenizer,
        trajectory_features_statistics=trajectory_stats,
        bounding_boxes=bounding_boxes,
        grid_name=config['grid_name'],
        component_configs=swipe_feature_extractor_config
    )

    dataset = SwipeDataset(
        data_path=config['data_path'],
        word_tokenizer=subword_tokenizer,
        grid_name_to_swipe_feature_extractor={config['grid_name']: feature_extractor},
        use_serialized_list=False
    )
    dataset_subset = SwipeDatasetSubset(dataset, config['grid_name'])

    # Load all component configs
    input_embedding_config = read_json(config['swipe_point_embedder_config_path'])
    encoder_config = read_json(config['encoder_config_path'])
    decoder_config = read_json(config['decoder_config_path'])

    model = get_model_from_configs(
        input_embedding_config=input_embedding_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_classes=config['num_classes'],
        n_word_tokens=len(subword_tokenizer.char_to_idx),
        max_out_seq_len=config['max_out_seq_len'],
        device=device,
        weights_path=config['model_weights_path']
    )
    model.eval()

    if num_workers > 1:
        model.share_memory()
        dataset_subset.dataset.data_list = manager.list(dataset_subset.dataset.data_list)
        dataset_subset.grid_idxs = manager.list(dataset_subset.grid_idxs)

    logit_processor = None
    if config.get('use_vocab_for_generation', False):
        vocab = get_vocab(config['voc_path'])
        logit_processor = VocabularyLogitProcessor(
            tokenizer=subword_tokenizer,
            vocab=vocab,
            max_token_id=config['num_classes'] - 1
        )
        if num_workers > 1:
            logit_processor.prefix_to_allowed_ids = manager.dict(logit_processor.prefix_to_allowed_ids)

    word_generator = generator_factory(
        generator_name=config['generator']['name'],
        model=model,
        subword_tokenizer=subword_tokenizer,
        device=device,
        logit_processor=logit_processor,
        generator_params=config['generator']['params']
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

    result = PredictionResult(predictions=all_predictions, config=config)
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    with open(config['output_path'], 'wb') as f:
        pickle.dump(result, f)

    print(f"Predictions saved to {config['output_path']}")


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    predict(config, args.num_workers)


if __name__ == "__main__":
    main()