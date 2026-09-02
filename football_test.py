"""
Football 网络谱半径置换检验诊断。

用途：解释 network/football.txt 为什么在基于谱半径的单侧检验中
得到 p=1，并生成零模型分布可视化。
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "centrality_test_mpl_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from centrality_test import sample_null_metrics, test_hub
from real_network_test import _from_edge_list


ROOT = Path(__file__).resolve().parent
FOOTBALL_PATH = ROOT / "network" / "football.txt"
DEFAULT_OUTPUT = ROOT / "football_pvalue_diagnostic.png"


def degree_stats(G: nx.Graph) -> dict[str, float]:
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    return {
        "min": float(np.min(degrees)),
        "max": float(np.max(degrees)),
        "mean": float(np.mean(degrees)),
        "std": float(np.std(degrees)),
        "cv": float(np.std(degrees) / np.mean(degrees)),
    }


def summarize_null(observed: float, null_values: np.ndarray) -> dict[str, float]:
    valid = null_values[~np.isnan(null_values)]
    ge_count = int(np.sum(valid >= observed))
    return {
        "observed": float(observed),
        "null_min": float(np.min(valid)),
        "null_mean": float(np.mean(valid)),
        "null_median": float(np.median(valid)),
        "null_max": float(np.max(valid)),
        "ge_count": ge_count,
        "n_valid": int(len(valid)),
        "p_value": float(ge_count / len(valid)),
    }


def plot_diagnostic(
    G: nx.Graph,
    observed: float,
    null_values: np.ndarray,
    model: str,
    n_samples: int,
    output_path: Path,
) -> Path:
    valid = null_values[~np.isnan(null_values)]
    stats = summarize_null(observed, valid)
    degrees = np.array([d for _, d in G.degree()], dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    axes[0].hist(valid, bins=30, color="#4C78A8", alpha=0.82, edgecolor="white")
    axes[0].axvline(observed, color="#D62728", linestyle="--", linewidth=2.4)
    axes[0].axvline(stats["null_mean"], color="#2CA02C", linestyle=":", linewidth=2.2)
    axes[0].set_title("Spectral radius: observed vs null")
    axes[0].set_xlabel("spectral_radius")
    axes[0].set_ylabel("Frequency")
    axes[0].text(
        0.02,
        0.98,
        (
            f"observed = {observed:.4f}\n"
            f"null min = {stats['null_min']:.4f}\n"
            f"null mean = {stats['null_mean']:.4f}\n"
            f"P(null >= observed) = {stats['p_value']:.4f}"
        ),
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BDBDBD"},
    )

    bins = np.arange(degrees.min() - 0.5, degrees.max() + 1.5, 1)
    axes[1].hist(degrees, bins=bins, color="#F58518", alpha=0.86, edgecolor="white")
    axes[1].set_title("Football degree distribution")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Node count")
    axes[1].set_xticks(np.arange(degrees.min(), degrees.max() + 1, 1))
    axes[1].text(
        0.02,
        0.98,
        (
            f"N = {G.number_of_nodes()}, M = {G.number_of_edges()}\n"
            f"degree range = {degrees.min()}..{degrees.max()}\n"
            f"degree CV = {np.std(degrees) / np.mean(degrees):.4f}"
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BDBDBD"},
    )

    fig.suptitle(f"football.txt hub test diagnostic ({model}, n={n_samples})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose football.txt hub-test p-value.")
    parser.add_argument(
        "--model",
        default="gnm",
        choices=["gnm", "gnp", "hidden", "configuration", "configuration_fast"],
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    G = _from_edge_list(FOOTBALL_PATH)
    print(f"football: N={G.number_of_nodes()}, M={G.number_of_edges()}")
    ds = degree_stats(G)
    print(
        "degree: "
        f"min={ds['min']:.0f}, max={ds['max']:.0f}, "
        f"mean={ds['mean']:.4f}, std={ds['std']:.4f}, cv={ds['cv']:.4f}"
    )
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = test_hub(
            G,
            n_samples=args.n_samples,
            model=args.model,
            alpha=args.alpha,
            seed=args.seed,
        )
        observed, null_records = sample_null_metrics(
            G,
            n_samples=args.n_samples,
            model=args.model,
            seed=args.seed,
        )

    print(result.to_string(index=False))
    print()

    metric = "spectral_radius"
    obs_val = observed[metric]
    null_values = np.array(null_records[metric], dtype=float)
    stats = summarize_null(obs_val, null_values)
    print(
        "null spectral_radius: "
        f"min={stats['null_min']:.6f}, median={stats['null_median']:.6f}, "
        f"mean={stats['null_mean']:.6f}, max={stats['null_max']:.6f}"
    )
    print(
        "p-value explanation: "
        f"{stats['ge_count']}/{stats['n_valid']} null samples have "
        f"spectral_radius >= observed, so p={stats['p_value']:.4f}."
    )
    if stats["ge_count"] == stats["n_valid"]:
        print("All sampled null graphs are at least as large as the observed graph on this metric.")

    fig_path = plot_diagnostic(
        G,
        observed=obs_val,
        null_values=null_values,
        model=args.model,
        n_samples=args.n_samples,
        output_path=args.output,
    )
    print(f"figure saved: {fig_path}")


if __name__ == "__main__":
    main()
