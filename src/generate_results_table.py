"""
Example usage:
python src/generate_results_table.py \
    --csv ./results/evaluation_results.csv \
    --sort_by mmr \
    --compare traj_and_weighted indiswipe \
    --experiment_names_mapping ./configs/report_experiment_names.json
"""

import argparse
from pathlib import Path
import json
from typing import Optional

import pandas as pd
from tabulate import tabulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate markdown tables from evaluation results CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to evaluation_results.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output markdown file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="mmr",
        choices=["mmr", "accuracy"],
        help="Metric to use for sorting experiments in the best performance table (default: mmr).",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        default=None,
        help="Generate comparison table between two experiment names (e.g., traj_and_weighted indiswipe).",
    )
    parser.add_argument(
        "--experiment_names_mapping",
        type=str,
        default=None,
        help="Optional JSON file mapping experiment names to display names.",
    )
    return parser.parse_args()


def load_experiment_mapping(mapping_path: Optional[str] = None) -> dict[str, str]:
    if mapping_path is None:
        return {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_best_performance(df: pd.DataFrame, sort_by: str = "mmr") -> pd.DataFrame:
    """Find the best performing epoch for each experiment."""
    results = []
    
    for exp_name in df["experiment_name"].unique():
        exp_df = df[df["experiment_name"] == exp_name]
        
        best_mmr_idx = exp_df["mmr"].idxmax()
        best_mmr_row = exp_df.loc[best_mmr_idx]
        
        best_acc_idx = exp_df["accuracy"].idxmax()
        best_acc_row = exp_df.loc[best_acc_idx]
        
        results.append({
            "experiment_name": exp_name,
            "best_mmr": best_mmr_row["mmr"],
            "mmr_epoch": best_mmr_row["epoch"],
            "best_accuracy": best_acc_row["accuracy"],
            "accuracy_epoch": best_acc_row["epoch"],
            "max_epoch": exp_df["epoch"].max(),
        })
    
    result_df = pd.DataFrame(results)
    
    sort_col = "best_mmr" if sort_by == "mmr" else "best_accuracy"
    result_df = result_df.sort_values(sort_col, ascending=False)
    
    return result_df


def render_best_per_experiment_table(best_df: pd.DataFrame, 
                                     name_map: dict[str, str], 
                                     sort_by: str) -> str:
    """Generate summary table showing best performance per experiment."""
    rows = []
    for _, row in best_df.iterrows():
        exp_name = row["experiment_name"]
        display_name = name_map.get(exp_name, exp_name)
        
        rows.append([
            display_name,
            f"{row['best_mmr']:.4f}",
            f"{row['best_accuracy']:.4f}",
            int(row["mmr_epoch"]),
            int(row["accuracy_epoch"]),
            int(row["max_epoch"]),
        ])
    
    headers = [
        "Experiment Name",
        "MRR",
        "Accuracy",
        "MMR Epoch",
        "Accuracy Epoch",
        "Max Considered Epoch",
    ]
    
    return tabulate(rows, headers=headers, tablefmt="github")


def render_comparison_table(
    best_df: pd.DataFrame,
    exp1_name: str,
    exp2_name: str,
    name_map: dict[str, str]
) -> str:
    """Generate comparison table showing delta between two experiments."""
    exp1_row = best_df[best_df["experiment_name"] == exp1_name]
    exp2_row = best_df[best_df["experiment_name"] == exp2_name]
    
    if exp1_row.empty:
        raise ValueError(f"Experiment '{exp1_name}' not found in results.")
    if exp2_row.empty:
        raise ValueError(f"Experiment '{exp2_name}' not found in results.")
    
    exp1_row = exp1_row.iloc[0]
    exp2_row = exp2_row.iloc[0]
    
    mmr_delta = exp1_row["best_mmr"] - exp2_row["best_mmr"]
    acc_delta = exp1_row["best_accuracy"] - exp2_row["best_accuracy"]
    
    mmr_delta_pct = (mmr_delta / exp2_row["best_mmr"]) * 100
    acc_delta_pct = (acc_delta / exp2_row["best_accuracy"]) * 100
    
    rows = [
        [
            name_map.get(exp1_name, exp1_name),
            f"{exp1_row['best_mmr']:.4f}",
            f"{exp1_row['best_accuracy']:.4f}",
        ],
        [
            name_map.get(exp2_name, exp2_name),
            f"{exp2_row['best_mmr']:.4f}",
            f"{exp2_row['best_accuracy']:.4f}",
        ],
        [
            "**Δ**",
            f"**{mmr_delta_pct:.2f}%**",
            f"**{acc_delta_pct:.2f}%**",
        ],
    ]
    
    headers = ["Features Type", "Swipe MRR", "Accuracy"]
    
    return tabulate(rows, headers=headers, tablefmt="github")


BEST_PERFORMANCE_SECTION_TEMPLATE = "## Best Performance per Experiment\n\n{summary_table}"

COMPARISON_SECTION_TEMPLATE = "## Comparison\n\n{comparison_table}\n"


def emit_output(result: str, output_path: Optional[Path]) -> None:
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Tables saved to: {output_path}")
    else:
        print(result)


def main():
    args = parse_args()
    
    df = pd.read_csv(args.csv)
    
    name_map = load_experiment_mapping(args.experiment_names_mapping)
    
    best_df = get_best_performance(df, sort_by=args.sort_by)
    
    summary_table = render_best_per_experiment_table(best_df, name_map, args.sort_by)
    
    comparison_table = None
    if args.compare:
        exp1, exp2 = args.compare
        comparison_table = render_comparison_table(best_df, exp1, exp2, name_map)
    
    
    result = "\n\n".join([
        BEST_PERFORMANCE_SECTION_TEMPLATE.format(summary_table=summary_table),
        COMPARISON_SECTION_TEMPLATE.format(comparison_table=comparison_table) if comparison_table else ""
    ])
    
    emit_output(result, args.output)


if __name__ == "__main__":
    main()
