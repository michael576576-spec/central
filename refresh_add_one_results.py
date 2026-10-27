"""Refresh saved Monte Carlo results after adopting the add-one p-value.

The legacy detail files contain p-values rounded to four or more decimals.
Their sample counts are at most 1000, so the integer tail counts are uniquely
recoverable. This script stores those counts for repeatable future refreshes.
It reuses the original null draws and does not change recorded runtimes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parent / "experiment_results"


def refresh_detail(path: Path, sample_column: str) -> pd.DataFrame:
    detail = pd.read_csv(path)
    samples = detail[sample_column].to_numpy(dtype=int)
    if "extreme_count" not in detail:
        raw_counts = detail["p_value"].to_numpy(dtype=float) * samples
        counts = np.rint(raw_counts).astype(int)
        if not np.allclose(raw_counts, counts, atol=5e-4, rtol=0):
            raise ValueError(f"Cannot recover tail counts from {path}")
        detail.insert(detail.columns.get_loc("p_value"), "extreme_count", counts)
    counts = detail["extreme_count"].to_numpy(dtype=int)
    if np.any((counts < 0) | (counts > samples)):
        raise ValueError(f"Invalid tail counts in {path}")
    detail["p_value"] = (counts + 1) / (samples + 1)
    new_reject = detail["p_value"].to_numpy() < detail["alpha"].to_numpy()
    if not np.array_equal(new_reject, detail["reject_H0"].to_numpy(dtype=bool)):
        labels = detail.groupby("reject_H0")["conclusion"].first().to_dict() if "conclusion" in detail else {}
        detail["reject_H0"] = new_reject
        if labels:
            detail["conclusion"] = [labels[bool(value)] for value in new_reject]
    detail.to_csv(path, index=False, encoding="utf-8-sig")
    return detail


def refresh_grid(path: Path) -> None:
    detail = refresh_detail(path, "n_samples")
    summary_path = path.with_name(path.name.replace("detail_grid", "summary_grid"))
    metadata = ["dataset", "model", "n_samples", "alpha", "seed", "run_elapsed_sec", "metric"]
    summary = detail[metadata].drop_duplicates().reset_index(drop=True).copy()
    for column in ("TP", "FP", "TN", "FN", "precision", "recall", "F1"):
        summary[column] = np.nan
    for index, row in summary.iterrows():
        group = detail.loc[
            (detail["dataset"] == row["dataset"])
            & (detail["model"] == row["model"])
            & (detail["n_samples"] == row["n_samples"])
            & (detail["metric"] == row["metric"])
        ]
        truth = group["has_hub"].to_numpy(dtype=bool)
        reject = group["reject_H0"].to_numpy(dtype=bool)
        tp = int(np.sum(truth & reject))
        fp = int(np.sum(~truth & reject))
        tn = int(np.sum(~truth & ~reject))
        fn = int(np.sum(truth & ~reject))
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
        summary.loc[index, ["TP", "FP", "TN", "FN"]] = [tp, fp, tn, fn]
        summary.loc[index, ["precision", "recall", "F1"]] = np.round([precision, recall, f1], 3)
    summary[["TP", "FP", "TN", "FN"]] = summary[["TP", "FP", "TN", "FN"]].astype(int)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


def refresh_sweep(directory: Path) -> None:
    detail = refresh_detail(directory / "hub_weight_sweep_detail.csv", "null_samples")
    summary = (
        detail.groupby(["hub_weight", "n_hubs"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            p_mean=("p_value", "mean"),
            p_median=("p_value", "median"),
            p_min=("p_value", "min"),
            p_max=("p_value", "max"),
            observed_mean=("observed", "mean"),
            observed_median=("observed", "median"),
            max_degree_mean=("max_degree", "mean"),
            hub_degree_mean=("hub_degree_mean", "mean"),
        )
    )
    summary.to_csv(directory / "hub_weight_sweep_summary.csv", index=False, encoding="utf-8-sig")
    spearman = (
        summary.groupby("hub_weight")
        .apply(lambda group: group[["n_hubs", "p_mean"]].corr(method="spearman").iloc[0, 1], include_groups=False)
        .rename("spearman")
        .reset_index()
    )
    by_weight = (
        detail.groupby("hub_weight", as_index=False)
        .agg(p_mean=("p_value", "mean"), p_median=("p_value", "median"),
             p_min=("p_value", "min"), p_max=("p_value", "max"))
        .merge(spearman, on="hub_weight")
    )
    by_weight = by_weight[["hub_weight", "spearman", "p_mean", "p_median", "p_min", "p_max"]]
    by_weight.to_csv(directory / "hub_weight_sweep_by_weight.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    for path in sorted(RESULTS.glob("detail_grid*.csv")):
        refresh_grid(path)
    for directory in sorted(RESULTS.glob("hub_weight_sweep_*")):
        if (directory / "hub_weight_sweep_detail.csv").exists():
            refresh_sweep(directory)
    notice = (
        "Archived original run log: p-values below use the old count/B estimator.\n"
        "Use the corrected detail and summary CSV files; see ADD_ONE_CORRECTION.md.\n\n"
    )
    for path in RESULTS.glob("*_batch_experiment.log"):
        original = path.read_text(encoding="utf-8")
        if not original.startswith(notice):
            path.write_text(notice + original, encoding="utf-8")
    print("Refreshed detail, summary, and by-weight CSV files using saved tail counts.")


if __name__ == "__main__":
    main()
