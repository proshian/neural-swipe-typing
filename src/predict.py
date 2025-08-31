import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import torch
from tqdm import tqdm

from dataset import SwipeDataset, SwipeDatasetSubset
from feature_extraction.swipe_feature_extractor_factory import swipe_feature_extractor_factory
from logit_processors import VocabularyLogitProcessor
from model import get_transformer__from_spe_config__vn1
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
    return parser.parse_args()


def predict(config: dict):
    device = torch.device(config['device'])

    subword_tokenizer = CharLevelTokenizerv2(config['voc_path'])
    keyboard_tokenizer = KeyboardTokenizer(config['keyboard_tokenizer_path'])

    grid = json.load(open(config['grids_path']))[config['grid_name']]
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
        grid_name_to_swipe_feature_extractor={config['grid_name']: feature_extractor}
    )
    dataset_subset = SwipeDatasetSubset(dataset, config['grid_name'])
    
    model = get_transformer__from_spe_config__vn1(
        spe_config=read_json(config['swipe_point_embedder_config_path']),
        n_classes=config['num_classes'],
        n_word_tokens=len(subword_tokenizer.char_to_idx),
        max_out_seq_len=config['max_out_seq_len'],
        device=device,
        weights_path=config['model_weights_path']
    )
    model.eval()

    logit_processor = None
    if config.get('use_vocab_for_generation', False):
        vocab = get_vocab(config['voc_path'])
        logit_processor = VocabularyLogitProcessor(
            tokenizer=subword_tokenizer,
            vocab=vocab,
            max_token_id=config['num_classes'] - 1
        )

    word_generator = generator_factory(
        model, subword_tokenizer, device, logit_processor, 
        generator_name=config['generator']['name'], 
        generator_params=config['generator']['params'])

    all_predictions = []
    for i in tqdm(range(len(dataset_subset)), desc="Generating predictions"):
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
    predict(config)


if __name__ == '__main__':
    main()
