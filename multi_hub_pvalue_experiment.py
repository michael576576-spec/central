"""
Experiment: fixed N/M multi-hub graphs and spectral-radius p-values.

The experiment keeps node count N and edge count M fixed. The first k nodes are
treated as hubs by assigning them a larger sampling weight, then M edges are
sampled without replacement from all node pairs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from centrality_test import compute_spectral_radius, test_hub


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "experiment_results" / "multi_hub_pvalue"
DEFAULT_HUB_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
DEFAULT_FULL_HUB_OUTPUT_DIR = ROOT / "experiment_results" / "full_hub_count_seed42"
DEFAULT_FULL_HUB_AVG_OUTPUT_DIR = ROOT / "experiment_results" / "full_hub_count_10seed_avg"


def _comb2(n: int) -> int:
    return n * (n - 1) // 2


def make_weighted_hub_graph_fixed_nm(
    N: int,
    M: int,
    n_hubs: int,
    seed: int | None = None,
    hub_weight: float = 8.0,
) -> nx.Graph:
    """
    Build a fixed-N/M graph where the first n_hubs nodes are favored.

    Edges are sampled without replacement from all node pairs. A pair's
    sampling weight is the product of endpoint weights: hub nodes use
    hub_weight, non-hub nodes use 1.0. When n_hubs == N, all pairs have equal
    weight and the construction becomes an ordinary G(N, M) sample.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if M < 0 or M > _comb2(N):
        raise ValueError("M must be between 0 and N*(N-1)/2")
    if n_hubs < 1 or n_hubs > N:
        raise ValueError("n_hubs must satisfy 1 <= n_hubs <= N")
    if hub_weight <= 0:
        raise ValueError("hub_weight must be positive")

    rng = np.random.default_rng(seed)
    node_weights = np.ones(N, dtype=float)
    node_weights[:n_hubs] = hub_weight

    edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    edge_weights = np.array(
        [node_weights[u] * node_weights[v] for u, v in edges],
        dtype=float,
    )
    probabilities = edge_weights / edge_weights.sum()
    chosen = rng.choice(len(edges), size=M, replace=False, p=probabilities)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges[int(i)] for i in chosen)
    return G


def graph_summary(G: nx.Graph, n_hubs: int) -> dict[str, float]:
    degrees = dict(G.degree())
    hub_degrees = [degrees[i] for i in range(n_hubs)] if n_hubs > 0 else []
    return {
        "observed": compute_spectral_radius(G),
        "max_degree": max(degrees.values()) if degrees else 0,
        "mean_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "hub_degree_min": min(hub_degrees) if hub_degrees else 0,
        "hub_degree_mean": float(np.mean(hub_degrees)) if hub_degrees else 0.0,
        "hub_degree_max": max(hub_degrees) if hub_degrees else 0,
    }


def run_multi_hub_pvalue_experiment(
    N: int = 160,
    M: int = 320,
    hub_counts: Iterable[int] = DEFAULT_HUB_COUNTS,
    graph_repeats: int = 30,
    null_samples: int = 100,
    model: str = "gnm",
    alpha: float = 0.05,
    seed: int = 42,
    hub_weight: float = 3.0,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    make_plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the fixed-N/M multi-hub experiment and save detail/summary outputs.

    Returns
    -------
    detail_df, summary_df
    """
    hub_counts = tuple(hub_counts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []
    for n_hubs in hub_counts:
        for repeat in range(graph_repeats):
            graph_seed = int(rng.integers(0, 2**31))
            null_seed = int(rng.integers(0, 2**31))
            G = make_weighted_hub_graph_fixed_nm(
                N=N,
                M=M,
                n_hubs=n_hubs,
                seed=graph_seed,
                hub_weight=hub_weight,
            )
            result = test_hub(
                G,
                n_samples=null_samples,
                model=model,
                alpha=alpha,
                seed=null_seed,
            ).iloc[0]
            metrics = graph_summary(G, n_hubs=n_hubs)
            rows.append({
                "N": N,
                "M": M,
                "n_hubs": n_hubs,
                "repeat": repeat,
                "graph_seed": graph_seed,
                "null_seed": null_seed,
                "model": model,
                "null_samples": null_samples,
                "alpha": alpha,
                "hub_weight": hub_weight,
                "p_value": float(result["p_value"]),
                "reject_H0": bool(result["reject_H0"]),
                **metrics,
            })

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df
        .groupby("n_hubs", as_index=False)
        .agg(
            graphs=("repeat", "count"),
            p_mean=("p_value", "mean"),
            p_median=("p_value", "median"),
            p_min=("p_value", "min"),
            p_max=("p_value", "max"),
            observed_mean=("observed", "mean"),
            max_degree_mean=("max_degree", "mean"),
            hub_degree_mean=("hub_degree_mean", "mean"),
        )
    )

    detail_path = output_dir / "multi_hub_pvalue_detail.csv"
    summary_path = output_dir / "multi_hub_pvalue_summary.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if make_plot:
        plot_multi_hub_pvalues(detail_df, summary_df, output_dir)

    spearman = detail_df[["n_hubs", "p_value"]].corr(method="spearman").iloc[0, 1]
    print(f"detail saved: {detail_path}")
    print(f"summary saved: {summary_path}")
    print(f"Spearman corr(n_hubs, p_value): {spearman:.4f}")
    print(summary_df.to_string(index=False))
    return detail_df, summary_df


def run_full_hub_count_experiment(
    N: int = 160,
    M: int = 320,
    hub_counts: Iterable[int] | None = None,
    seed: int = 42,
    null_samples: int = 100,
    model: str = "gnm",
    alpha: float = 0.05,
    hub_weight: float = 8.0,
    output_dir: str | Path = DEFAULT_FULL_HUB_OUTPUT_DIR,
    make_plot: bool = True,
) -> pd.DataFrame:
    """
    Run one graph per hub count from 1..N with fixed N/M and a shared seed.
    """
    if hub_counts is None:
        hub_counts = range(1, N + 1)
    hub_counts = tuple(hub_counts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_hubs in hub_counts:
        G = make_weighted_hub_graph_fixed_nm(
            N=N,
            M=M,
            n_hubs=n_hubs,
            seed=seed,
            hub_weight=hub_weight,
        )
        result = test_hub(
            G,
            n_samples=null_samples,
            model=model,
            alpha=alpha,
            seed=seed,
        ).iloc[0]
        rows.append({
            "N": N,
            "M": M,
            "n_hubs": n_hubs,
            "seed": seed,
            "model": model,
            "null_samples": null_samples,
            "alpha": alpha,
            "hub_weight": hub_weight,
            "p_value": float(result["p_value"]),
            "reject_H0": bool(result["reject_H0"]),
            **graph_summary(G, n_hubs=n_hubs),
        })

    result_df = pd.DataFrame(rows)
    result_path = output_dir / "full_hub_count_pvalue.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    if make_plot:
        plot_full_hub_count_result(result_df, output_dir)

    spearman = result_df[["n_hubs", "p_value"]].corr(method="spearman").iloc[0, 1]
    print(f"result saved: {result_path}")
    print(f"Spearman corr(n_hubs, p_value): {spearman:.4f}")
    print(result_df[[
        "n_hubs",
        "p_value",
        "reject_H0",
        "observed",
        "max_degree",
        "hub_degree_mean",
    ]].to_string(index=False))
    return result_df


def run_full_hub_count_seed_average_experiment(
    N: int = 160,
    M: int = 320,
    hub_counts: Iterable[int] | None = None,
    seeds: Iterable[int] = tuple(range(42, 52)),
    null_samples: int = 100,
    model: str = "gnm",
    alpha: float = 0.05,
    hub_weight: float = 8.0,
    output_dir: str | Path = DEFAULT_FULL_HUB_AVG_OUTPUT_DIR,
    make_plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Average the full hub-count experiment over multiple graph seeds.
    """
    if hub_counts is None:
        hub_counts = range(1, N + 1)
    hub_counts = tuple(hub_counts)
    seeds = tuple(seeds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        for n_hubs in hub_counts:
            G = make_weighted_hub_graph_fixed_nm(
                N=N,
                M=M,
                n_hubs=n_hubs,
                seed=seed,
                hub_weight=hub_weight,
            )
            result = test_hub(
                G,
                n_samples=null_samples,
                model=model,
                alpha=alpha,
                seed=seed,
            ).iloc[0]
            rows.append({
                "N": N,
                "M": M,
                "n_hubs": n_hubs,
                "seed": seed,
                "model": model,
                "null_samples": null_samples,
                "alpha": alpha,
                "hub_weight": hub_weight,
                "p_value": float(result["p_value"]),
                "reject_H0": bool(result["reject_H0"]),
                **graph_summary(G, n_hubs=n_hubs),
            })

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df
        .groupby("n_hubs", as_index=False)
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

    n_seeds = len(seeds)
    detail_path = output_dir / f"full_hub_count_{n_seeds}seed_detail.csv"
    summary_path = output_dir / f"full_hub_count_{n_seeds}seed_summary.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if make_plot:
        plot_full_hub_count_average_result(detail_df, summary_df, output_dir, n_seeds=n_seeds)

    spearman = detail_df[["n_hubs", "p_value"]].corr(method="spearman").iloc[0, 1]
    print(f"detail saved: {detail_path}")
    print(f"summary saved: {summary_path}")
    print(f"Spearman corr(n_hubs, p_value): {spearman:.4f}")
    print(summary_df[[
        "n_hubs",
        "p_mean",
        "p_median",
        "observed_mean",
        "max_degree_mean",
        "hub_degree_mean",
    ]].to_string(index=False))
    return detail_df, summary_df


def run_hub_weight_sweep_experiment(
    N: int = 50,
    M: int = 100,
    hub_counts: Iterable[int] | None = None,
    seeds: Iterable[int] = tuple(range(42, 92)),
    hub_weights: Iterable[float] = (1.5, 2.0, 3.0, 4.0, 6.0),
    null_samples: int = 100,
    model: str = "gnm",
    alpha: float = 0.05,
    output_dir: str | Path | None = None,
    make_plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run multiple hub_weight values for the same fixed-N/M hub-count experiment.
    """
    if hub_counts is None:
        hub_counts = range(1, N + 1)
    hub_counts = tuple(hub_counts)
    seeds = tuple(seeds)
    hub_weights = tuple(float(w) for w in hub_weights)
    n_seeds = len(seeds)

    if output_dir is None:
        output_dir = (
            ROOT
            / "experiment_results"
            / f"hub_weight_sweep_N{N}_M{M}_{model}_{n_seeds}seed"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for hub_weight in hub_weights:
        for seed in seeds:
            for n_hubs in hub_counts:
                G = make_weighted_hub_graph_fixed_nm(
                    N=N,
                    M=M,
                    n_hubs=n_hubs,
                    seed=seed,
                    hub_weight=hub_weight,
                )
                result = test_hub(
                    G,
                    n_samples=null_samples,
                    model=model,
                    alpha=alpha,
                    seed=seed,
                ).iloc[0]
                rows.append({
                    "N": N,
                    "M": M,
                    "n_hubs": n_hubs,
                    "seed": seed,
                    "model": model,
                    "null_samples": null_samples,
                    "alpha": alpha,
                    "hub_weight": hub_weight,
                    "p_value": float(result["p_value"]),
                    "reject_H0": bool(result["reject_H0"]),
                    **graph_summary(G, n_hubs=n_hubs),
                })

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df
        .groupby(["hub_weight", "n_hubs"], as_index=False)
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
    weight_summary_df = (
        detail_df
        .groupby("hub_weight", as_index=False)
        .agg(
            spearman=(
                "p_value",
                lambda s: detail_df.loc[s.index, ["n_hubs", "p_value"]]
                .corr(method="spearman")
                .iloc[0, 1],
            ),
            p_mean=("p_value", "mean"),
            p_median=("p_value", "median"),
            p_min=("p_value", "min"),
            p_max=("p_value", "max"),
        )
    )

    detail_path = output_dir / "hub_weight_sweep_detail.csv"
    summary_path = output_dir / "hub_weight_sweep_summary.csv"
    weight_summary_path = output_dir / "hub_weight_sweep_by_weight.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    weight_summary_df.to_csv(weight_summary_path, index=False, encoding="utf-8-sig")

    if make_plot:
        plot_hub_weight_sweep_result(summary_df, output_dir, N=N, M=M, n_seeds=n_seeds, model=model)

    print(f"detail saved: {detail_path}")
    print(f"summary saved: {summary_path}")
    print(f"weight summary saved: {weight_summary_path}")
    print(weight_summary_df.to_string(index=False))
    return detail_df, summary_df, weight_summary_df


def plot_full_hub_count_result(
    result_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FULL_HUB_OUTPUT_DIR,
) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "centrality_test_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), dpi=150, sharex=True)
    axes[0].plot(result_df["n_hubs"], result_df["p_value"], linewidth=1.8)
    axes[0].axhline(0.05, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("p-value")
    axes[0].set_title("Fixed N=160, M=320, seed=42")

    axes[1].plot(result_df["n_hubs"], result_df["observed"], color="#D62728", linewidth=1.8)
    axes[1].set_ylabel("spectral radius")

    axes[2].plot(result_df["n_hubs"], result_df["max_degree"], color="#4C78A8", linewidth=1.8)
    axes[2].plot(
        result_df["n_hubs"],
        result_df["hub_degree_mean"],
        color="#2CA02C",
        linewidth=1.8,
        label="mean hub degree",
    )
    axes[2].set_xlabel("number of hub nodes")
    axes[2].set_ylabel("degree")
    axes[2].legend()

    fig.tight_layout()
    output_path = output_dir / "full_hub_count_pvalue.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_hub_weight_sweep_result(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    N: int,
    M: int,
    n_seeds: int,
    model: str,
) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "centrality_test_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    for hub_weight, grp in summary_df.groupby("hub_weight"):
        axes[0].plot(
            grp["n_hubs"],
            grp["p_mean"],
            linewidth=1.6,
            label=f"w={hub_weight:g}",
        )
        axes[1].plot(
            grp["n_hubs"],
            grp["observed_mean"],
            linewidth=1.6,
            label=f"w={hub_weight:g}",
        )

    axes[0].axhline(0.05, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("mean p-value")
    axes[0].set_title(f"{model} p-values by hub_weight (N={N}, M={M}, {n_seeds} seeds)")
    axes[0].legend()

    axes[1].set_xlabel("number of hub nodes")
    axes[1].set_ylabel("mean spectral radius")
    axes[1].legend()

    fig.tight_layout()
    output_path = output_dir / "hub_weight_sweep_p_mean.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_full_hub_count_average_result(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FULL_HUB_AVG_OUTPUT_DIR,
    n_seeds: int = 10,
) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "centrality_test_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), dpi=150, sharex=True)
    axes[0].plot(
        summary_df["n_hubs"],
        summary_df["p_mean"],
        color="#D62728",
        linewidth=1.8,
        label="mean p",
    )
    axes[0].plot(
        summary_df["n_hubs"],
        summary_df["p_median"],
        color="#9467BD",
        linewidth=1.4,
        label="median p",
    )
    axes[0].axhline(0.05, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("p-value")
    N = int(detail_df["N"].iloc[0])
    M = int(detail_df["M"].iloc[0])
    axes[0].set_title(f"Fixed N={N}, M={M}, averaged over {n_seeds} seeds")
    axes[0].legend()

    axes[1].plot(
        summary_df["n_hubs"],
        summary_df["max_degree_mean"],
        color="#4C78A8",
        linewidth=1.8,
        label="mean max degree",
    )
    axes[1].plot(
        summary_df["n_hubs"],
        summary_df["hub_degree_mean"],
        color="#F58518",
        linewidth=1.8,
        label="mean hub degree",
    )
    axes[1].set_ylabel("degree")
    axes[1].legend()

    axes[2].plot(
        summary_df["n_hubs"],
        summary_df["observed_mean"],
        color="#2CA02C",
        linewidth=1.8,
        label="mean spectral radius",
    )
    axes[2].set_xlabel("number of hub nodes")
    axes[2].set_ylabel("spectral radius")
    axes[2].legend()

    fig.tight_layout()
    output_path = output_dir / f"full_hub_count_{n_seeds}seed_avg.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_multi_hub_pvalues(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "centrality_test_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    hub_counts = summary_df["n_hubs"].tolist()
    p_groups = [
        detail_df.loc[detail_df["n_hubs"] == k, "p_value"].to_numpy()
        for k in hub_counts
    ]

    axes[0].boxplot(
        p_groups,
        tick_labels=[str(k) for k in hub_counts],
        showfliers=False,
    )
    axes[0].plot(
        range(1, len(hub_counts) + 1),
        summary_df["p_median"],
        color="#D62728",
        marker="o",
        linewidth=1.8,
        label="median p",
    )
    axes[0].axhline(0.05, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xlabel("number of hub nodes")
    axes[0].set_ylabel("p-value")
    axes[0].set_title("p-value by hub count")
    axes[0].legend()

    axes[1].plot(
        summary_df["n_hubs"],
        summary_df["observed_mean"],
        marker="o",
        color="#4C78A8",
        linewidth=1.8,
    )
    axes[1].set_xlabel("number of hub nodes")
    axes[1].set_ylabel("mean spectral radius")
    axes[1].set_title("Observed spectral radius")

    fig.tight_layout()
    output_path = output_dir / "multi_hub_pvalue.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    run_hub_weight_sweep_experiment(
        N=50,
        M=100,
        hub_counts=range(1, 51),
        seeds=range(42, 92),
        hub_weights=(1.5, 2.0, 3.0, 4.0, 6.0),
        null_samples=100,
        model="gnm",
        output_dir=ROOT / "experiment_results" / "hub_weight_sweep_N50_M100_K1_50_gnm_50seed",
        make_plot=True,
    )
