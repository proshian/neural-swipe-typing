import sys; import os; sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "src"))


import json
import os
from pathlib import Path
from typing import List
import logging

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from lightning.pytorch import loggers as pl_loggers
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from dataset import CollateFn, SwipeDataset, SwipeDatasetSubset
from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizer
from feature_extraction.swipe_feature_extractor_factory import swipe_feature_extractor_factory
from feature_extraction.swipe_feature_extractors import MultiFeatureExtractor, TrajectoryFeatureExtractor
from pl_module import LitNeuroswipeModel
from train_utils import CrossEntropyLossWithReshape
from model import get_model_from_configs
from utils.get_git_commit_hash import get_git_commit_hash


logger = logging.getLogger(__name__)


def get_config_derived_name(cfg: DictConfig) -> str:
    """
    Generate experiment name from resolved config's type fields and CLI overrides.

    Uses the 'type' field from each component config to build a descriptive name,
    then appends any CLI overrides for full reproducibility visibility.

    Example output:
    trajectory+nearest_key__traj_and_nearest__conformer__transformer_v1__encoder.params.dropout=0.2
    """
    encoder_name = cfg.encoder.type
    decoder_name = cfg.decoder.type
    spe_name = cfg.swipe_point_embedder.type.replace("separate_", "")

    feature_extractor_types = [fe.type for fe in cfg.feature_extractor]
    fe_name = "+".join(feature_extractor_types)

    base_name = f"{fe_name}__{spe_name}__{encoder_name}__{decoder_name}"

    # Get CLI overrides from Hydra and append them to the name
    overrides = HydraConfig.get().overrides.task
    if overrides:
        # Sort for consistent ordering
        overrides_str = "__".join(sorted(overrides))
        return f"{base_name}__{overrides_str}"

    return base_name


def _setup_logging(logging_level: str = "INFO") -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=getattr(logging, logging_level))


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj


def get_n_traj_feats(feature_extractor: MultiFeatureExtractor) -> int:
    # ! Note: There is an alternative to just call traj_feat_extractor
    # on sample data and return the shape of the output.
    traj_feat_extractor = None
    for feature_extractor_component in feature_extractor.extractors:
        if isinstance(feature_extractor_component, TrajectoryFeatureExtractor):
            traj_feat_extractor = feature_extractor_component
            break
    if traj_feat_extractor is None:
        return 0
    N_COORD_FEATS = 2  # x and y
    n_traj_feats = (N_COORD_FEATS
                    + traj_feat_extractor.include_dt
                    + 2*traj_feat_extractor.include_velocities
                    + 2*traj_feat_extractor.include_accelerations)
    return n_traj_feats


def validate_d_model(
    d_model_config: int,
    feature_extractor: MultiFeatureExtractor,
    input_embedding_config: dict
) -> int:
    """
    Validate d_model consistency across configs.

    Arguments:
    ----------
    d_model_config: int
        The d_model value specified in the training configuration
    feature_extractor: MultiFeatureExtractor
        Feature extractor used to compute trajectory features count
    input_embedding_config: dict
        Swipe point embedder configuration containing key_emb_size

    Returns:
    --------
    d_model: int
        The validated d_model value

    Raises:
    -------
    ValueError
        If d_model is not specified in config or doesn't match expected value
    """
    # Compute expected d_model from feature extractor and swipe point embedder configs
    n_coord_feats = get_n_traj_feats(feature_extractor)
    key_emb_size = input_embedding_config["params"]["key_emb_size"]
    expected_d_model = n_coord_feats + key_emb_size

    if d_model_config != expected_d_model:
        raise ValueError(
            f"d_model mismatch: config specifies d_model={d_model_config}, "
            f"but feature extractor and swipe point embedder produce "
            f"{n_coord_feats} (trajectory) + {key_emb_size} (key embedding) = {expected_d_model}"
        )
    logger.info(f"d_model={d_model_config} (trajectory feats: {n_coord_feats}, key_emb_size: {key_emb_size})")


def create_lr_scheduler_ctor(scheduler_type: str, scheduler_params: dict):
    def get_lr_scheduler(optimizer):
        if scheduler_type == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unknown lr_scheduler type: {scheduler_type}")
    return get_lr_scheduler


def create_optimizer_ctor(optimizer_type: str, optimizer_kwargs: dict, no_decay_keys: List[str] = None):
    """
    Create optimizer constructor with configurable weight decay exclusion.
    """
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    if weight_decay > 0 and no_decay_keys is None:
        raise ValueError(
            "no_decay_keys must be explicitly set when weight_decay > 0. "
            "Use [] to apply weight decay to all parameters, or specify parameter name substrings "
            "to exclude (examples: `['bias', 'LayerNorm.weight', 'norm.weight']`, `[]`)."
        )

    if no_decay_keys is None:
        no_decay_keys = []

    def get_optimizer(model_named_parameters):
        decay_params = []
        no_decay_params = []

        for name, param in model_named_parameters:
            if any(nd_key in name for nd_key in no_decay_keys):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer_kwargs_without_weight_decay = {
            k: v for k, v in optimizer_kwargs.items() if k != "weight_decay"
        }

        optimizer_classes = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD
        }

        if optimizer_type not in optimizer_classes:
            raise ValueError(
                f"Unknown optimizer type: {optimizer_type}. "
                f"Supported types: {list(optimizer_classes.keys())}"
            )

        optimizer_class = optimizer_classes[optimizer_type]
        return optimizer_class(optimizer_grouped_parameters, **optimizer_kwargs_without_weight_decay)

    return get_optimizer


def get_callbacks(experiment_name: str, early_stopping_config: dict) -> List[Callback]:
    # Sanitize experiment name for filename (replace / with --)
    ckpt_name = experiment_name.replace('/', '--')
    ckpt_filename = f'{ckpt_name}-{{epoch}}-{{val_loss:.4f}}-{{val_word_level_accuracy:.4f}}'

    model_checkpoint_top = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=10,
        dirpath=f'checkpoints/{experiment_name}/top_10', filename=ckpt_filename)

    model_checkpoint_on_epoch_end = ModelCheckpoint(
        save_on_train_epoch_end=True, dirpath=f'checkpoints/{experiment_name}/epoch_end/',
        save_top_k=-1,
        filename=ckpt_filename)

    callbacks = [
        model_checkpoint_top,
        model_checkpoint_on_epoch_end,
    ]

    if early_stopping_config["enabled"]:
        early_stopping_cb = EarlyStopping(
            monitor='val_loss', mode='min',
            patience=early_stopping_config["patience"])
        callbacks.append(early_stopping_cb)

    return callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    original_cwd = hydra.utils.get_original_cwd()

    def resolve_path(path: str) -> str:
        if Path(path).is_absolute():
            return path
        return str(Path(original_cwd) / path)

    logging_level = cfg.get("logging_level", "INFO")
    _setup_logging(logging_level)

    # Print the resolved config
    logger.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

    # Convert Hydra configs to dicts for factory functions
    encoder_config = OmegaConf.to_container(cfg.encoder, resolve=True)
    decoder_config = OmegaConf.to_container(cfg.decoder, resolve=True)
    swipe_point_embedder_config = OmegaConf.to_container(cfg.swipe_point_embedder, resolve=True)
    feature_extractor_config = OmegaConf.to_container(cfg.feature_extractor, resolve=True)

    grids = read_json(resolve_path(cfg.grids_path))
    grid = grids[cfg.grid_name]
    trajectory_features_statistics = read_json(resolve_path(cfg.trajectory_features_statistics_path))
    bounding_boxes = read_json(resolve_path(cfg.bounding_boxes_path))

    keyboard_tokenizer = KeyboardTokenizer(resolve_path(cfg.keyboard_tokenizer_path))
    word_tokenizer = CharLevelTokenizerv2(resolve_path(cfg.vocab_path))
    word_pad_idx = word_tokenizer.char_to_idx['<pad>']

    config_name = get_config_derived_name(cfg)
    default_experiment_name = config_name
    experiment_name = cfg.get("experiment_name", default_experiment_name)

    # Assertions
    assert 1 <= cfg.num_classes <= len(word_tokenizer.char_to_idx), \
        "num_classes should be between 1 and the number of tokens in the vocabulary"

    path_to_continue_checkpoint = cfg.get("path_to_continue_checkpoint", None)

    seed_everything(cfg.seed)

    feature_extractor = swipe_feature_extractor_factory(
        grid, keyboard_tokenizer, trajectory_features_statistics,
        bounding_boxes, cfg.grid_name, feature_extractor_config)

    grid_name_to_swipe_feature_extractor = {cfg.grid_name: feature_extractor}

    persistent_workers = cfg.dataloader_num_workers > 0

    train_dataset_full = SwipeDataset(
        data_path=resolve_path(cfg.dataset_paths.train),
        word_tokenizer=word_tokenizer,
        grid_name_to_swipe_feature_extractor=grid_name_to_swipe_feature_extractor,
        total=cfg.get("train_total", None)
    )
    train_dataset = SwipeDatasetSubset(train_dataset_full, grid_name=cfg.grid_name)

    val_dataset_full = SwipeDataset(
        data_path=resolve_path(cfg.dataset_paths.val),
        word_tokenizer=word_tokenizer,
        grid_name_to_swipe_feature_extractor=grid_name_to_swipe_feature_extractor,
        total=cfg.get("val_total", None)
    )
    val_dataset = SwipeDatasetSubset(val_dataset_full, grid_name=cfg.grid_name)

    collate_fn = CollateFn(word_pad_idx=word_pad_idx, batch_first=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn)

    log_dir = resolve_path(cfg.get("log_dir", "lightning_logs"))
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    # Save configs and metadata for reproducibility
    os.makedirs(tb_logger.log_dir, exist_ok=True)

    # Save as YAML (Hydra native)
    config_yaml_path = os.path.join(tb_logger.log_dir, "config.yaml")
    OmegaConf.save(cfg, config_yaml_path)

    # Save as JSON for compatibility with analysis scripts
    config_json_path = os.path.join(tb_logger.log_dir, "config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4, ensure_ascii=False)

    # Save git commit hash
    commit_hash = get_git_commit_hash() or "git commit hash unavailable"
    with open(os.path.join(tb_logger.log_dir, "git_commit_hash.txt"), "w", encoding="utf-8") as f:
        f.write(commit_hash + "\n")

    callbacks = get_callbacks(experiment_name, cfg.early_stopping)

    criterion = CrossEntropyLossWithReshape(
        ignore_index=word_pad_idx,
        label_smoothing=cfg.get("label_smoothing", 0.0))

    lr_scheduler_ctor = create_lr_scheduler_ctor(
        cfg.lr_scheduler.type,
        OmegaConf.to_container(cfg.lr_scheduler.params, resolve=True)
    )

    optimizer_ctor = create_optimizer_ctor(
        cfg.optimizer.type,
        OmegaConf.to_container(cfg.optimizer.params, resolve=True),
        no_decay_keys=cfg.optimizer.get("no_decay_keys", None)
    )

    # Validate d_model
    d_model = cfg.get("d_model")
    if d_model is None:
        raise ValueError(
            "d_model must be specified in train config. "
            "It should match the output dimension of the swipe point embedder."
        )
    validate_d_model(d_model, feature_extractor, swipe_point_embedder_config)

    model = get_model_from_configs(
        input_embedding_config=swipe_point_embedder_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_classes=cfg.num_classes,
        n_word_tokens=len(word_tokenizer.char_to_idx),
        max_out_seq_len=cfg.max_out_seq_len,
        d_model=d_model,
        device=cfg.device
    )

    pl_model = LitNeuroswipeModel(
        model=model,
        criterion=criterion,
        word_pad_idx=word_pad_idx,
        num_classes=cfg.num_classes,
        train_batch_size=cfg.train_batch_size,
        optimizer_ctor=optimizer_ctor,
        lr_scheduler_ctor=lr_scheduler_ctor,
    )

    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    trainer = Trainer(
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        accelerator='gpu',
        precision=cfg.trainer_precision,
        callbacks=callbacks,
        logger=tb_logger,
        val_check_interval=cfg.val_check_interval
    )

    trainer.fit(
        pl_model, train_loader, val_loader,
        ckpt_path=path_to_continue_checkpoint
    )


if __name__ == "__main__":
    main()
