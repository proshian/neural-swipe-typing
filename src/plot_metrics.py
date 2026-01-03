import os
import argparse
import json
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="CSV file from evaluate.py")
    parser.add_argument("--metrics", nargs="+", required=True, help="List of metric names to plot")
    parser.add_argument("--output_dir", type=str, default="plots", help="Where to save plots")
    parser.add_argument("--colors_config", type=str, default=None, help="Optional JSON config for experiment colors")
    return parser.parse_args()


def get_colors_config(colors_config_path: Optional[str]) -> Optional[dict]:
    if colors_config_path is None:
        return None
    with open(colors_config_path, 'r') as f:
        return json.load(f)


def plot_metrics(csv_path, metrics, output_dir="plots", colors_config=None):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    experiments = df["experiment_name"].unique()

    for metric in metrics:
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in CSV, skipping.")
            continue

        plt.figure()
        for experiment in experiments:
            experiment_df = df[df["experiment_name"] == experiment].sort_values("epoch")
            plt.plot(experiment_df["epoch"], experiment_df[metric], label=experiment,
                     color=colors_config.get(experiment) if colors_config else None)

        plt.title(f"{metric} vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

if __name__ == "__main__":
    args = parse_args()
    colors_config = get_colors_config(args.colors_config)
    plot_metrics(args.csv, args.metrics, args.output_dir, colors_config)