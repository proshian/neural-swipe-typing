import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot metrics from TensorBoard lightning_logs by experiment."
    )
    parser.add_argument(
        "--tb_logdir_root",
        type=str,
        required=True,
        help="Path to lightning_logs root containing $experiment_name/version_* subdirs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Scalar tags to plot (e.g., train_loss_epoch, val_word_level_accuracy). If omitted, plots all available metrics.",
    )
    parser.add_argument(
        "--colors_config",
        type=str,
        default=None,
        help="Optional JSON mapping of experiment_name -> color hex.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots_tb",
        help="Directory to save output PNGs.",
    )
    return parser.parse_args()


def load_colors(colors_path: Optional[str]) -> dict[str, str]:
    if not colors_path:
        return {}
    with open(colors_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_version_dirs(experiment_dir: Path) -> list[Path]:
    dirs = []
    for child in experiment_dir.iterdir():
        if child.is_dir() and child.name.startswith("version_"):
            idx_str = child.name.replace("version_", "")
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            dirs.append((idx, child))
    dirs.sort(key=lambda x: x[0])
    return [d for _, d in dirs]


def collect_scalars(tb_root: Path, metrics: Optional[list[str]]) -> pd.DataFrame:
    """
    Collect scalar metrics from TensorBoard logs for specified metrics.
    If metrics is None, collects all available scalars.

    The resulting DataFrame has columns: experiment, metric, step, value.
    Later versions overwrite earlier versions for the same step.
    It's supposed that the versions are subsequent runs of the same experiment
    that continue training from previous checkpoints.
    """
    rows: list[dict[str, Any]] = []

    for experiment_dir in sorted(tb_root.iterdir()):
        if not experiment_dir.is_dir():
            continue

        versions = get_version_dirs(experiment_dir)
        if not versions:
            continue

        for version_dir in versions:
            acc = EventAccumulator(str(version_dir))
            acc.Reload()
            all_scalar_tags = set(acc.Tags().get("scalars", []))

            metrics_to_collect = metrics if metrics is not None else all_scalar_tags

            for metric in metrics_to_collect:
                if metric not in all_scalar_tags:
                    print(f"Warning: Metric '{metric}' not found in {experiment_dir.name}/{version_dir.name}; skipping.")
                    continue

                scalars = acc.Scalars(metric)
                if not scalars:
                    continue

                for s in scalars:
                    rows.append(
                        {
                            "experiment": experiment_dir.name,
                            "metric": metric,
                            "step": s.step,
                            "value": s.value,
                        }
                    )

    df = pd.DataFrame(rows)
    # Keep last occurrence (later versions overwrite earlier ones)
    df = df.drop_duplicates(subset=["experiment", "metric", "step"], keep="last")
    return df


def plot_metrics(df: pd.DataFrame, colors: dict[str, str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for metric in df["metric"].unique():
        metric_df = df[df["metric"] == metric]

        plt.figure()
        for experiment, group in metric_df.groupby("experiment"):
            group_sorted = group.sort_values("step")
            plt.plot(
                group_sorted["step"],
                group_sorted["value"],
                label=experiment,
                color=colors.get(experiment),
            )

        plt.title(f"{metric} vs Steps")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")


def main():
    args = parse_args()
    tb_root = Path(args.tb_logdir_root)

    if not tb_root.exists():
        raise FileNotFoundError(f"tb_logdir_root does not exist: {tb_root}")

    colors = load_colors(args.colors_config)
    df = collect_scalars(tb_root, args.metrics)

    if df.empty:
        print("No scalar data found. Check paths, metrics, and logs.")
        return

    plot_metrics(df, colors, args.output_dir)


if __name__ == "__main__":
    main()
