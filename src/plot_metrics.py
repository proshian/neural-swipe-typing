import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="CSV file from evaluate.py")
    parser.add_argument("--metrics", nargs="+", required=True, help="List of metric names to plot")
    parser.add_argument("--output_dir", type=str, default="plots", help="Where to save plots")
    return parser.parse_args()


def plot_metrics(csv_path, metrics, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    experiments = df["experiment_name"].unique()

    for metric in metrics:
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in CSV, skipping.")
            continue

        plt.figure()
        for experiment in experiments:
            experiment_df = df[df["experiment_name"] == experiment]
            plt.plot(experiment_df["epoch"], experiment_df[metric], marker="o", label=experiment)

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
    plot_metrics(args.csv, args.metrics, args.output_dir)
