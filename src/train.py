import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))


import json
import os
import argparse
from typing import List
import logging

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from lightning.pytorch import loggers as pl_loggers

from dataset import CollateFn, SwipeDataset, SwipeDatasetSubset
from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizer
from feature_extraction.swipe_feature_extractor_factory import swipe_feature_extractor_factory
from feature_extraction.swipe_feature_extractors import MultiFeatureExtractor, TrajectoryFeatureExtractor
from pl_module import LitNeuroswipeModel
from train_utils import CrossEntropyLossWithReshape
from train_utils import EmptyCudaCacheCallback
from model import get_transformer__from_spe_config__vn1


logger = logging.getLogger(__name__)


LOG_DIR = "lightning_logs/"



def _setup_logging(train_config: dict) -> None:
    """Configure logging, even if called after logger creation."""
    # Remove all handlers from root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging_level_str = train_config.get("logging_level", "INFO")
    logging.basicConfig(level=getattr(logging, logging_level_str))
    


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)  
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train.json",
        help="Path to the training configuration file",
    )
    args = parser.parse_args()
    return args


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
    
    Arguments:
    ----------
    optimizer_type: str
        Type of optimizer ('Adam', 'AdamW', 'SGD')
    optimizer_kwargs: dict
        Optimizer keyword arguments
    no_decay_keys: list | None
        List of parameter name substrings to exclude from weight decay.
        Must be explicitly set when weight_decay > 0.
        Use [] to apply weight decay to all parameters.
        Examples: 
        * ['bias', 'LayerNorm.weight', 'norm.weight']
        * []
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



def get_callbacks(train_config) -> List[Callback]:
    grid_name = train_config["grid_name"]
    ckpt_filename = (f'{train_config["model_name"]}-{grid_name}--' 
                     + '{epoch}-{val_loss:.3f}-{val_word_level_accuracy:.3f}')

    model_checkpoint_top = ModelCheckpoint(
        monitor='val_loss', mode = 'min', save_top_k=10,
        dirpath=f'checkpoints/{train_config["experiment_name"]}/top_10', filename=ckpt_filename)

    model_checkpoint_on_epoch_end = ModelCheckpoint(
        save_on_train_epoch_end = True, dirpath=f'checkpoints/{train_config["experiment_name"]}/epoch_end/',
        save_top_k=-1,
        filename=ckpt_filename)
    
    # When num workers > 0, there's a RAM drain issue:
    # See: https://github.com/pytorch/pytorch/issues/13246.
    # emptying cuda cache is a workaround.
    # However, it doesn't seem to work with pytorch lightning,
    # (the callback doesn't solve an issue, but is kept as a reminder)
    callbacks = [
        model_checkpoint_top, 
        model_checkpoint_on_epoch_end,
        EmptyCudaCacheCallback()
    ]
    
    if train_config["early_stopping"]["enabled"]:
        early_stopping_cb = EarlyStopping(
            monitor='val_loss', mode = 'min', 
            patience=train_config["early_stopping"]["patience"])
        callbacks.append(early_stopping_cb)
    
    return callbacks
    
    

def main(train_config: dict) -> None:
    # Read components from config
    grid_name = train_config["grid_name"]
    trajectory_features_statistics = read_json(train_config["trajectory_features_statistics_path"])        
    bounding_boxes = read_json(train_config["bounding_boxes_path"])
    grids = read_json(train_config["grids_path"])
    grid = grids[grid_name]
    swipe_feature_extractor_component_configs = read_json(train_config["swipe_feature_extractor_factory_config_path"])
    keyboard_tokenizer = KeyboardTokenizer(train_config["keyboard_tokenizer_path"])
    persistent_workers = True if train_config["dataloader_num_workers"] > 0 else False
    word_tokenizer = CharLevelTokenizerv2(train_config["vocab_path"])
    feature_extractor_name = os.path.basename(train_config["trajectory_features_statistics_path"]).split(".")[0]
    default_experiment_name = f"{train_config['model_name']}__{grid_name}__{feature_extractor_name}__bs_{train_config['train_batch_size']}/seed_{train_config['seed']}"
    experiment_name = train_config.get("experiment_name", default_experiment_name)
    word_pad_idx = word_tokenizer.char_to_idx['<pad>']

    # Assertions
    # num_classes is usually equal to num_tokens - 2 (subtracting <pad> and <unk>)
    assert 1 <= train_config["num_classes"] <= len(word_tokenizer.char_to_idx), \
        "num_classes should be between 1 and the number of tokens in the vocabulary"


    path_to_continue_checkpoint = None
    if train_config.get("path_to_continue_checkpoint", None):
        path_to_continue_checkpoint = train_config["path_to_continue_checkpoint"]


    seed_everything(train_config["seed"])


    feature_extractor = swipe_feature_extractor_factory(
        grid, keyboard_tokenizer, trajectory_features_statistics, 
        bounding_boxes, grid_name, swipe_feature_extractor_component_configs)

    grid_name_to_swipe_feature_extractor = {
        grid_name: feature_extractor
    }


    train_dataset_full = SwipeDataset(
        data_path=train_config["dataset_paths"]["train"],
        word_tokenizer=word_tokenizer,
        grid_name_to_swipe_feature_extractor=grid_name_to_swipe_feature_extractor,
        total=train_config.get("train_total", None)
    )
    train_dataset = SwipeDatasetSubset(train_dataset_full, grid_name=grid_name)

    val_dataset_full = SwipeDataset(
        data_path=train_config["dataset_paths"]["val"],
        word_tokenizer=word_tokenizer,
        grid_name_to_swipe_feature_extractor=grid_name_to_swipe_feature_extractor,
        total=train_config.get("val_total", None)
    )
    val_dataset = SwipeDatasetSubset(val_dataset_full, grid_name=grid_name)


    collate_fn = CollateFn(
        word_pad_idx=word_pad_idx, batch_first=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config["train_batch_size"], 
        shuffle=True,
        num_workers=train_config["dataloader_num_workers"],
        persistent_workers = persistent_workers,
        collate_fn=collate_fn)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_config["val_batch_size"], 
        shuffle=False,
        num_workers=train_config["dataloader_num_workers"], 
        persistent_workers = persistent_workers,
        collate_fn=collate_fn)
    


    tb_logger = pl_loggers.TensorBoardLogger(save_dir=LOG_DIR, name=experiment_name)

    # Save train config to the logger directory
    # for reproducibility.
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    with open(os.path.join(tb_logger.log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=4, ensure_ascii=False)
    

    callbacks = get_callbacks(train_config)


    criterion = CrossEntropyLossWithReshape(
        ignore_index=word_pad_idx, 
        label_smoothing=train_config.get("label_smoothing", 0.0))
    

    lr_scheduler_ctor=create_lr_scheduler_ctor(
        train_config["lr_scheduler"]["type"],
        train_config["lr_scheduler"]["params"]
    )

    optimizer_ctor = create_optimizer_ctor(
        train_config["optimizer"]["type"],
        train_config["optimizer"]["params"],
        no_decay_keys=train_config["optimizer"].get("no_decay_keys", None)
    )

    spe_config = read_json(train_config["swipe_point_embedder_config_path"])
    model = get_transformer__from_spe_config__vn1(
        spe_config=spe_config,
        n_classes= train_config["num_classes"],
        n_word_tokens=len(word_tokenizer.char_to_idx),
        max_out_seq_len=train_config["max_out_seq_len"],
        device=train_config["device"]
    )

    # n_coord_feats = get_n_traj_feats(feature_extractor)
    # key_emb_size = train_config["swipe_point_embedder_config"]["params"]["key_emb_size"]
    # TODO add assert that d_model == n_coord_feats + key_emb_size

    pl_model = LitNeuroswipeModel(
        model=model,
        criterion = criterion, 
        word_pad_idx = word_pad_idx,
        num_classes = train_config["num_classes"],
        train_batch_size = train_config["train_batch_size"],
        optimizer_ctor=optimizer_ctor, 
        lr_scheduler_ctor=lr_scheduler_ctor, 
    )

    trainer = Trainer(
    #     limit_train_batches = 400,  # for validating code before actual training
        log_every_n_steps = 100,
        num_sanity_val_steps=0,
        accelerator = 'gpu',
        # max_epochs=100,
        callbacks=callbacks,
        logger=tb_logger,
        val_check_interval=train_config["val_check_interval"]
    )

    trainer.fit(
        pl_model, train_loader, val_loader,
        ckpt_path = path_to_continue_checkpoint
    )


    
if __name__ == "__main__":
    args = parse_args()
    train_config = read_json(args.train_config)
    _setup_logging(train_config)
    main(train_config)
