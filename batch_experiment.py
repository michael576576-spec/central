"""
Batch runner for real and simulated hub-detection experiments.

It evaluates both data sources over a grid of null models and sample counts,
and writes one log file per null model.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Iterable

import pandas as pd

import real_network_test
import simulate_graph_test


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "experiment_results"
DEFAULT_MODELS = ("gnm", "configuration", "hidden")
DEFAULT_SAMPLE_COUNTS = (100, 500, 1000)
DEFAULT_DATASETS = ("real", "simulated")


def _evaluate_dataset(
    dataset: str,
    n_samples: int,
    model: str,
    alpha: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "real":
        return real_network_test.evaluate(
            n_samples=n_samples,
            model=model,
            alpha=alpha,
            seed=seed,
            verbose=False,
        )
    if dataset == "simulated":
        return simulate_graph_test.evaluate(
            n_samples=n_samples,
            model=model,
            alpha=alpha,
            seed=seed,
            verbose=False,
        )
    raise ValueError(f"unknown dataset: {dataset!r}")


def run_model_sample_grid(
    models: Iterable[str] = DEFAULT_MODELS,
    sample_counts: Iterable[int] = DEFAULT_SAMPLE_COUNTS,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    alpha: float = 0.05,
    seed: int = 42,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """
    Run experiments for each dataset, model, and n_samples combination.

    Returns paths for the generated per-model logs and CSV files.

    Per-graph progress logging is intentionally disabled so log files only
    contain complete result blocks.
    """
    models = tuple(models)
    sample_counts = tuple(sample_counts)
    datasets = tuple(datasets)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_grid.csv"
    detail_path = output_dir / "detail_grid.csv"
    log_paths = {model: output_dir / f"{model}_batch_experiment.log" for model in models}

    all_summaries: list[pd.DataFrame] = []
    all_details: list[pd.DataFrame] = []

    for model in models:
        model_started_at = time.perf_counter()
        log_path = log_paths[model]
        with log_path.open("w", encoding="utf-8") as log_file:
            print("Batch experiment started", file=log_file)
            print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}", file=log_file)
            print(f"model: {model}", file=log_file)
            print(f"datasets: {list(datasets)}", file=log_file)
            print(f"sample_counts: {list(sample_counts)}", file=log_file)
            print(f"alpha: {alpha}", file=log_file)
            print(f"seed: {seed}", file=log_file)
            print(file=log_file)

            for n_samples in sample_counts:
                for dataset in datasets:
                    # Each result block starts with its full experiment key.
                    # Per-graph progress from evaluate(verbose=True) is omitted.
                    combo_started_at = time.perf_counter()
                    print("=" * 80, file=log_file)
                    print(
                        f"dataset={dataset}, model={model}, "
                        f"n_samples={n_samples}, alpha={alpha}, seed={seed}",
                        file=log_file,
                    )
                    print("=" * 80, file=log_file)

                    summary_df, detail_df = _evaluate_dataset(
                        dataset=dataset,
                        n_samples=n_samples,
                        model=model,
                        alpha=alpha,
                        seed=seed,
                    )

                    elapsed_sec = time.perf_counter() - combo_started_at

                    summary_df = summary_df.copy()
                    detail_df = detail_df.copy()
                    for df in (summary_df, detail_df):
                        df.insert(0, "dataset", dataset)
                        df.insert(1, "model", model)
                        df.insert(2, "n_samples", n_samples)
                        df.insert(3, "alpha", alpha)
                        df.insert(4, "seed", seed)
                        df.insert(5, "run_elapsed_sec", round(elapsed_sec, 4))

                    all_summaries.append(summary_df)
                    all_details.append(detail_df)

                    print(file=log_file)
                    print("summary:", file=log_file)
                    print(summary_df.to_string(index=False), file=log_file)
                    print(file=log_file)
                    print("detail:", file=log_file)
                    print(detail_df.to_string(index=False), file=log_file)
                    print(file=log_file)
                    print(f"combo elapsed_sec={elapsed_sec:.4f}", file=log_file)
                    print(file=log_file)
                    log_file.flush()

            print("=" * 80, file=log_file)
            print(
                f"Model experiment finished, "
                f"elapsed_sec={time.perf_counter() - model_started_at:.4f}",
                file=log_file,
            )
            print(f"summary_csv: {summary_path}", file=log_file)
            print(f"detail_csv: {detail_path}", file=log_file)

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(summary_path, index=False, encoding="utf-8-sig")

    if all_details:
        pd.concat(all_details, ignore_index=True).to_csv(
            detail_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(detail_path, index=False, encoding="utf-8-sig")

    return {
        "logs": log_paths,
        "summary_csv": summary_path,
        "detail_csv": detail_path,
    }


if __name__ == "__main__":
    paths = run_model_sample_grid(models=("configuration_fast",));
    for model, log_path in paths["logs"].items():
        print(f"{model} log saved: {log_path}")
    print(f"summary saved: {paths['summary_csv']}")
    print(f"detail saved: {paths['detail_csv']}")
