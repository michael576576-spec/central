"""Generate component figures for the spectral-radius hub test.

The script creates two deliberately contrasting graphs with the same ``N`` and
``M``: a hub-dominated positive example and a representative negative example.
Each graph is tested against 500 GNM (G(N, M)) random graphs, so the
figures can be placed directly into a workflow diagram.

Run from the repository root::

    python generate_gnm_components.py

Outputs are written to ``experiment_results/gnm_components`` by default.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

# Keep matplotlib's cache inside the writable workspace temp area when the
# user's global cache directory is unavailable (common in managed runtimes).
_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "centrality_test_mpl_cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from centrality_test import compute_spectral_radius, sample_null_metrics


N_NODES = 30
N_EDGES = 50
N_SAMPLES = 500
ALPHA = 0.05


def make_positive_graph(n: int = N_NODES, m: int = N_EDGES) -> nx.Graph:
    """Create a connected graph with a moderate, statistically detectable hub."""
    if m < n - 1 or m >= n * (n - 1) // 2:
        raise ValueError("m must allow a simple connected graph")

    g = nx.Graph()
    g.add_nodes_from(range(n))
    hub = 0
    # Give the hub a noticeable but non-extreme neighborhood.
    for node in range(1, min(16, n)):
        g.add_edge(hub, node)
    # Add a simple backbone through the remaining nodes so the graph stays
    # connected without turning the hub into a star center.
    for node in range(15, n - 1):
        g.add_edge(node, node + 1)
    rng = np.random.default_rng(20260917)
    all_edges = [(u, v) for u in range(1, n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    for edge in all_edges:
        if g.number_of_edges() >= m:
            break
        g.add_edge(*edge)
    return g


def connected_gnm_graph(n: int, m: int, seed: int | None = None) -> nx.Graph:
    """Sample G(N,M) until a connected graph is obtained for visualization."""
    rng = np.random.default_rng(seed)
    while True:
        graph = nx.gnm_random_graph(n, m, seed=int(rng.integers(2**31)))
        if nx.is_connected(graph):
            return graph


def p_value(observed: float, null_values: list[float]) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0 or np.isnan(observed):
        return float("nan")
    return float(np.mean(values >= observed))


def select_mid_null_graph(
    n: int,
    m: int,
    n_samples: int,
    seed: int,
    candidates: int = 160,
) -> nx.Graph:
    """Pick a GNM graph whose independent-null p-value is near 0.5.

    This keeps the negative example representative rather than making it an
    unusually low-spectral-radius graph with p close to 1.0.
    """
    reference = connected_gnm_graph(n, m, seed=seed)
    _, reference_records = sample_null_metrics(
        reference, n_samples=n_samples, model="gnm", seed=seed + 1
    )
    reference_null = reference_records["spectral_radius"]

    best_graph = reference
    best_distance = float("inf")
    for offset in range(candidates):
        graph = connected_gnm_graph(n, m, seed=seed + 2 + offset)
        candidate_p = p_value(compute_spectral_radius(graph), reference_null)
        distance = abs(candidate_p - 0.5)
        if distance < best_distance:
            best_graph = graph
            best_distance = distance
    return best_graph


def draw_network(g: nx.Graph, path: Path, title: str, p_val: float) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.5), dpi=180)
    pos = nx.spring_layout(g, seed=12, k=0.7, iterations=200)
    degrees = np.asarray([g.degree[node] for node in g], dtype=float)
    sizes = 38 + 16 * degrees
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#A7B1BE", width=0.9, alpha=0.8)
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=degrees,
        cmap="viridis",
        node_size=sizes,
        linewidths=0.7,
        edgecolors="white",
    )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.text(
        0.02,
        0.02,
        f"N={g.number_of_nodes()}, M={g.number_of_edges()}\n"
        f"spectral radius={compute_spectral_radius(g):.3f}\n"
        f"GNM samples={N_SAMPLES}, p={p_val:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#D0D7DE"},
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_distribution(
    observed: float,
    null_values: list[float],
    path: Path,
    title: str,
    p_val: float,
) -> None:
    values = np.asarray(null_values, dtype=float)
    values = values[~np.isnan(values)]
    fig, ax = plt.subplots(figsize=(5.4, 3.7), dpi=180)
    ax.hist(values, bins=28, color="#5B8FF9", alpha=0.82, edgecolor="white")
    ax.axvline(observed, color="#D64545", linestyle="--", linewidth=2.0)
    ax.text(
        0.98,
        0.95,
        f"observed={observed:.3f}\np={p_val:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#D0D7DE"},
    )
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Spectral radius")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_summary(results: list[dict[str, object]], path: Path, n_samples: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    for row, result in enumerate(results):
        g = result["graph"]
        pos = nx.spring_layout(g, seed=12, k=0.7, iterations=200)
        degrees = np.asarray([g.degree[node] for node in g], dtype=float)
        nx.draw_networkx_edges(g, pos, ax=axes[row, 0], edge_color="#A7B1BE", width=0.8)
        nx.draw_networkx_nodes(
            g, pos, ax=axes[row, 0], node_color=degrees, cmap="viridis",
            node_size=34 + 14 * degrees, edgecolors="white", linewidths=0.6,
        )
        axes[row, 0].set_title(str(result["label"]), weight="bold")
        axes[row, 0].axis("off")

        values = np.asarray(result["null_values"], dtype=float)
        axes[row, 1].hist(values[~np.isnan(values)], bins=28, color="#5B8FF9", alpha=0.82, edgecolor="white")
        axes[row, 1].axvline(float(result["observed"]), color="#D64545", linestyle="--", linewidth=2)
        axes[row, 1].set_title(f"GNM null distribution ({n_samples} samples), p={float(result['p_value']):.3f}")
        axes[row, 1].set_xlabel("Spectral radius")
        axes[row, 1].set_ylabel("Count")
        axes[row, 1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Spectral-radius hub test: positive vs negative examples", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results/gnm_components"))
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    negative = select_mid_null_graph(N_NODES, N_EDGES, args.samples, args.seed + 100)
    examples = [("Positive example", make_positive_graph()), ("Negative example", negative)]
    results: list[dict[str, object]] = []
    for index, (label, graph) in enumerate(examples):
        observed, null_records = sample_null_metrics(
            graph, n_samples=args.samples, model="gnm", seed=args.seed + index
        )
        observed_value = float(observed["spectral_radius"])
        null_values = null_records["spectral_radius"]
        p_val = p_value(observed_value, null_values)
        results.append({
            "label": label,
            "graph": graph,
            "observed": observed_value,
            "null_values": null_values,
            "p_value": p_val,
            "reject_H0": p_val < ALPHA,
        })

        stem = "positive" if index == 0 else "negative"
        draw_network(graph, args.output_dir / f"gnm_{stem}_network.png", label, p_val)
        draw_distribution(
            observed_value,
            null_values,
            args.output_dir / f"gnm_{stem}_null_distribution.png",
            f"{label}: GNM null distribution",
            p_val,
        )

    draw_summary(results, args.output_dir / "gnm_positive_negative_summary.png", args.samples)
    with (args.output_dir / "gnm_examples_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "nodes", "edges", "spectral_radius", "p_value", "reject_H0"])
        writer.writeheader()
        for result in results:
            graph = result["graph"]
            writer.writerow({
                "label": result["label"],
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "spectral_radius": f"{float(result['observed']):.6f}",
                "p_value": f"{float(result['p_value']):.6f}",
                "reject_H0": bool(result["reject_H0"]),
            })
    print(f"Wrote component figures and summary to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
