import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import argparse
import json
import os
import pickle
import re
from typing import List, Tuple, Dict, Optional

import pandas as pd
from tqdm import tqdm

from predict import PredictionResult
from metrics import get_mmr, get_accuracy

COLUMN_ORDER = [
    "experiment_name",
    "epoch",
    "mmr",
    "accuracy",
    "grid_name",
    "generator",
    "use_vocab_for_generation",
    "data_split",
    "model_weights_file",
]

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
    # Check using all columns except metrics to ensure complete configuration match
    METRIC_COLUMNS = ['accuracy', 'mmr']
    config_columns = [col for col in COLUMN_ORDER if col not in METRIC_COLUMNS]
    result_series = pd.Series({col: str(result.get(col)) for col in config_columns})
    return (df[config_columns].astype(str) == result_series).all(axis=1).any()


def get_result_dict(prediction_result: PredictionResult, metrics_dict: Optional[Dict[str, float]] = None) -> dict:
    """Extracts the result dictionary from a prediction without computing metrics."""
    config = prediction_result.config
    generator_config = json.dumps(config['generator'])
    
    experiment_configuration_dict = {
        "experiment_name": config["experiment_name"],
        "epoch": extract_epoch_from_path(config["model_weights_path"]),
        "grid_name": config["grid_name"],
        "generator": generator_config,
        "use_vocab_for_generation": config["use_vocab_for_generation"],
        "data_split": os.path.basename(config["data_path"]),
        "model_weights_file": os.path.basename(config["model_weights_path"]),
    }

    metrics_dict = metrics_dict or {"accuracy": None, "mmr": None}

    return {**experiment_configuration_dict, **metrics_dict}


def save_results(prediction_result: PredictionResult, metrics: Dict[str, float], out_path: str) -> None:
    """
    Appends the evaluation result of a single prediction file to the output CSV.
    Ensures consistent column order using COLUMN_ORDER.
    """
    result_dict = get_result_dict(prediction_result, metrics)
    df_line = pd.DataFrame([result_dict])[COLUMN_ORDER]
    
    if not os.path.exists(out_path):
        df_line.to_csv(out_path, index=False)
        return
    
    existing_df = pd.read_csv(out_path).reindex(columns=COLUMN_ORDER)
    
    if not is_result_in_df(existing_df, result_dict):
        combined_df = pd.concat([existing_df, df_line], ignore_index=True)
        combined_df.to_csv(out_path, index=False)


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
    labels = get_labels_from_ds_path(data_path, [grid_name])
    preds = scored_preds_to_raw_preds(prediction_result.predictions)

    assert len(preds) == len(labels), f"Length mismatch: {len(preds)} vs {len(labels)}"

    mmr = get_mmr(cut_inner_lists_to_four(preds), labels)
    accuracy = get_accuracy(leave_one_pred_per_curve(preds), labels)
    metrics = {'mmr': mmr, 'accuracy': accuracy}

    save_results(prediction_result, metrics, config['output_csv_path'])


def filter_already_evaluated_files(prediction_file_paths: List[str], output_csv_path: str) -> List[str]:
    """Filters out prediction files that have already been evaluated."""
    if not os.path.exists(output_csv_path):
        return prediction_file_paths
    
    existing_results_df = pd.read_csv(output_csv_path)
    existing_results_df = existing_results_df.reindex(columns=COLUMN_ORDER, fill_value=pd.NA)
    
    unevaluated_files = []
    
    for prediction_path in tqdm(prediction_file_paths, desc="Checking for already evaluated files"):
        prediction_result = read_prediction(prediction_path)
        result_dict = get_result_dict(prediction_result, metrics_dict=None)
        
        if not is_result_in_df(existing_results_df, result_dict):
            unevaluated_files.append(prediction_path)

    return unevaluated_files


if __name__ == "__main__":
    config = get_config()
    prediction_file_paths = find_prediction_files(config['prediction_paths'])
    
    if not prediction_file_paths:
        raise FileNotFoundError("No .pkl files found in specified prediction_paths. Exiting.")
    
    os.makedirs(os.path.dirname(config['output_csv_path']), exist_ok=True)
    
    unevaluated_files = filter_already_evaluated_files(prediction_file_paths, config['output_csv_path'])
    
    print(f"Found {len(prediction_file_paths)} prediction files, {len(unevaluated_files)} need evaluation")
    
    for prediction_path in tqdm(unevaluated_files, desc="Evaluating prediction files"):
        evaluate(prediction_path, config)

    print(f"\nEvaluation finished. Results saved to {config['output_csv_path']}")
