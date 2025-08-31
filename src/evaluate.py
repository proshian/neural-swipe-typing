import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import argparse
import json
import os
import pickle
import re
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

from predict import PredictionResult
from metrics import get_mmr, get_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True, help="Path to the evaluation config file.")
    args = p.parse_args()
    return args


def get_config() -> dict:
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    return config


def read_prediction(prediction_path: str) -> PredictionResult:
    with open(prediction_path, 'rb') as f:
        prediction = pickle.load(f)
    return prediction


def get_labels_from_ds_path(dataset_path: str, 
                            gnames_to_include: List[str]
                            ) -> List[str]:
    labels = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line)
            gname = line_data['curve']['grid_name']
            if gname in gnames_to_include:
                labels.append(line_data['word'])
    return labels


def scored_preds_to_raw_preds(scored_preds: List[List[Tuple[float, str]]]
                              ) -> List[List[str]]:
    preds_only = []
    for scored_preds_for_curve_i in scored_preds:
        preds_only.append([pred for score, pred in scored_preds_for_curve_i])
    return preds_only


def extract_epoch_from_path(path: str) -> int:
    match = re.search(r'epoch=(\d+)', path)
    if match:
        return int(match.group(1))
    return -1


def is_result_in_df(df: pd.DataFrame, result: dict) -> bool:
    result_df = pd.DataFrame([result])
    merged_df = df.merge(result_df, on=list(df.columns), how='inner')
    return not merged_df.empty


def save_results(prediction_result: PredictionResult, metrics: Dict[str, float], out_path: str) -> None:
    """
    Appends the evaluation result of a single prediction file to the output CSV.
    Checks for duplicates before writing.
    """
    config = prediction_result.config

    generator_config = json.dumps(config['generator'])
    
    result_dict = {
        "experiment_name": config.get("experiment_name"),
        "epoch": extract_epoch_from_path(config.get("model_weights_path", "")),
        "mmr": metrics.get('mmr'),
        "accuracy": metrics.get('accuracy'),
        "grid_name": config.get("grid_name"),
        "generator": generator_config,
        "use_vocab_for_generation": config["use_vocab_for_generation"],
        "data_split": os.path.basename(config["data_path"]),
        "model_weights_file": os.path.basename(config.get("model_weights_path", "")),
    }

    df_line = pd.DataFrame([result_dict])

    if not os.path.exists(out_path):
        df_line.to_csv(out_path, index=False)
    else:
        df = pd.read_csv(out_path)
        # Convert columns to object type to avoid dtype issues during check
        if not is_result_in_df(df.astype(object), result_dict):
            df_line.to_csv(out_path, mode='a', header=False, index=False)


def find_prediction_files(prediction_paths: List[str]) -> List[str]:
    """Recursively finds all prediction files from the configured paths."""
    prediction_file_paths = []
    for path_str in prediction_paths:
        if os.path.isfile(path_str):
            prediction_file_paths.append(path_str)
        elif os.path.isdir(path_str):
            for root, _, files in os.walk(path_str):
                for file in files:
                    prediction_file_paths.append(os.path.join(root, file))
    return prediction_file_paths


def cut_inner_lists_to_four(preds):
    return [preds_for_curve[:4] for preds_for_curve in preds]


def leave_one_pred_per_curve(preds):
    return [preds_for_curve[0] for preds_for_curve in preds]


def evaluate(prediction_path: str, config: dict) -> None:
    prediction_result = read_prediction(prediction_path)
    grid_name = prediction_result.config.get("grid_name")
    data_path = prediction_result.config['data_path']
    labels = get_labels_from_ds_path(data_path, grid_name)
    preds = scored_preds_to_raw_preds(prediction_result.predictions)

    assert len(preds) == len(labels), f"Length mismatch: {len(preds)} vs {len(labels)}"

    mmr = get_mmr(cut_inner_lists_to_four(preds), labels)
    accuracy = get_accuracy(leave_one_pred_per_curve(preds), labels)
    metrics = {'mmr': mmr, 'accuracy': accuracy}

    save_results(prediction_result, metrics, config['output_csv_path'])


if __name__ == "__main__":
    config = get_config()
    prediction_file_paths = find_prediction_files(config['prediction_paths'])
    
    if not prediction_file_paths:
        raise FileNotFoundError("No .pkl files found in specified prediction_paths. Exiting.")
    
    os.makedirs(os.path.dirname(config['output_csv_path']), exist_ok=True)
    
    for prediction_path in tqdm(prediction_file_paths, desc="Evaluating prediction files"):
        evaluate(prediction_path, config)

    print(f"\nEvaluation finished. Results saved to {config['output_csv_path']}")
