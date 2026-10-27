"""
Evaluate the detection power of the hub-structure test.

Use synthetic graphs with known labels (positive = hub structure,
negative = no hub structure) to assess precision, recall, and F1 for
the metrics in centrality_test.test_hub.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import networkx as nx

from centrality_test import test_hub


# ─────────────────────────────────────────────
# Build synthetic graphs
# ─────────────────────────────────────────────

def _hub_er(n_total: int, hub_degree_ratio: float, seed: int) -> nx.Graph:
    """
    Graph with one hub and an ER background.
    Node 0 connects to floor(n_total * hub_degree_ratio) random nodes.
    Other pairs connect with p = 2*M_background / (N*(N-1)).
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_total))

    n_hub_edges = max(1, int(n_total * hub_degree_ratio))
    hub_neighbors = rng.choice(range(1, n_total), size=min(n_hub_edges, n_total - 1), replace=False)
    for v in hub_neighbors:
        G.add_edge(0, v)

    # Background edge density is about half the hub-edge density.
    p_bg = n_hub_edges / (n_total * (n_total - 1) / 2) * 0.5
    for u in range(1, n_total):
        for v in range(u + 1, n_total):
            if rng.random() < p_bg:
                G.add_edge(u, v)
    return G


def _double_hub(n_total: int, hub_degree: int, seed: int) -> nx.Graph:
    """
    Two hubs (nodes 0 and 1), each connected to hub_degree distinct leaves.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_total))

    leaves = list(range(2, n_total))
    rng.shuffle(leaves)
    half = min(hub_degree, len(leaves) // 2)
    for v in leaves[:half]:
        G.add_edge(0, v)
    for v in leaves[half: half * 2]:
        G.add_edge(1, v)
    return G


def _ba_with_explicit_hub(
    n_total: int,
    m: int,
    hub_degree_ratio: float,
    seed: int,
) -> nx.Graph:
    """
    BA background with an explicit strong hub.

    A plain BA graph can be borderline. Add edges from node 0 to a large
    share of nodes to make the result a clear positive example.
    """
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n_total, m, seed=seed)

    n_hub_edges = max(1, int(n_total * hub_degree_ratio))
    current_neighbors = set(G.neighbors(0))
    candidates = [v for v in range(1, n_total) if v not in current_neighbors]
    rng.shuffle(candidates)

    n_to_add = min(max(0, n_hub_edges - len(current_neighbors)), len(candidates))
    for v in candidates[:n_to_add]:
        G.add_edge(0, v)
    return G


def make_benchmark_graphs(seed: int = 42) -> list[dict]:
    """
    Build synthetic graphs with known hub-structure labels.

    Returns
    -------
    List of dicts, each containing:
        G           : nx.Graph
        has_hub     : bool (True = positive, False = negative)
        graph_type  : str (descriptive label)
        N           : int
    """
    rng = np.random.default_rng(seed)
    entries = []

    def child_seed():
        return int(rng.integers(0, 2**31))

    # Positive examples: graphs with hub structure.
    # Star graph with an extreme hub.
    for n_leaves in [19, 29, 49, 79, 119]:
        entries.append({
            "G": nx.star_graph(n_leaves),
            "has_hub": True,
            "graph_type": f"star_N{n_leaves + 1}",
            "N": n_leaves + 1,
        })

    # Hub with an ER background.
    for N, ratio in [
        (40, 0.45),
        (40, 0.65),
        (60, 0.40),
        (60, 0.60),
        (80, 0.45),
        (80, 0.70),
        (100, 0.50),
        (100, 0.75),
    ]:
        G = _hub_er(N, ratio, seed=child_seed())
        entries.append({
            "G": G,
            "has_hub": True,
            "graph_type": f"hub_er_N{N}_ratio{ratio}",
            "N": N,
        })

    # Two-hub graph.
    for N, hub_degree in [(50, 18), (70, 25), (100, 35), (140, 50)]:
        G = _double_hub(N, hub_degree=hub_degree, seed=child_seed())
        entries.append({
            "G": G,
            "has_hub": True,
            "graph_type": f"double_hub_N{N}_d{hub_degree}",
            "N": N,
        })

    # BA graph with a strong hub added to the background.
    for N, m, ratio in [(60, 2, 0.45), (100, 2, 0.50), (140, 3, 0.55)]:
        G = _ba_with_explicit_hub(N, m, ratio, seed=child_seed())
        entries.append({
            "G": G,
            "has_hub": True,
            "graph_type": f"ba_explicit_hub_N{N}_m{m}_ratio{ratio}",
            "N": N,
        })

    # Negative examples: graphs without hub structure.
    # ER random graph with uniform edge probabilities.
    for N, p in [
        (40, 0.12),
        (50, 0.10),
        (60, 0.09),
        (80, 0.075),
        (80, 0.12),
        (100, 0.06),
        (120, 0.055),
        (120, 0.08),
        (150, 0.045),
        (200, 0.035),
    ]:
        G = nx.erdos_renyi_graph(N, p, seed=child_seed())
        entries.append({
            "G": G,
            "has_hub": False,
            "graph_type": f"er_N{N}_p{p}",
            "N": N,
        })

    # Regular graph: all nodes have equal degree, a clear negative example.
    for N, d in [
        (40, 4),
        (50, 6),
        (60, 4),
        (80, 6),
        (100, 4),
        (100, 8),
        (120, 6),
        (150, 4),
        (200, 6),
        (240, 8),
    ]:
        G = nx.random_regular_graph(d, N, seed=child_seed())
        entries.append({
            "G": G,
            "has_hub": False,
            "graph_type": f"regular_N{N}_d{d}",
            "N": N,
        })

    return entries


# ─────────────────────────────────────────────
# Test one graph
# ─────────────────────────────────────────────

def run_one(
    entry: dict,
    n_samples: int,
    model: str,
    alpha: float,
    seed: int,
) -> list[dict]:
    """
    Run test_hub on one graph and return a row per metric.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_df = test_hub(entry["G"], n_samples=n_samples, model=model,
                             alpha=alpha, seed=seed)

    rows = []
    for _, row in result_df.iterrows():
        rows.append({
            "graph_type": entry["graph_type"],
            "N":          entry["N"],
            "M":          entry["G"].number_of_edges(),
            "has_hub":    entry["has_hub"],
            "metric":     row["metric"],
            "observed":   row["observed"],
            "p_value":    row["p_value"],
            "reject_H0":  row["reject_H0"],
            "conclusion": row["conclusion"],
        })
    return rows


# ─────────────────────────────────────────────
# Aggregate evaluation
# ─────────────────────────────────────────────

def evaluate(
    n_samples: int = 500,
    model: str = "hidden",
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test all synthetic graphs and calculate precision, recall, and F1.

    Returns
    -------
    summary_df : Per-metric TP, FP, TN, FN, precision, recall, and F1.
    detail_df  : Per-graph, per-metric results.
    """
    graphs = make_benchmark_graphs(seed=seed)
    rng = np.random.default_rng(seed + 1)
    child_seeds = rng.integers(0, 2**31, size=len(graphs)).tolist()

    all_rows = []
    for i, entry in enumerate(graphs):
        if verbose:
            print(f"  [{i+1}/{len(graphs)}] {entry['graph_type']} (has_hub={entry['has_hub']})")
        rows = run_one(entry, n_samples=n_samples, model=model, alpha=alpha,
                       seed=child_seeds[i])
        all_rows.extend(rows)

    detail_df = pd.DataFrame(all_rows)

    # Count true positives, false positives, true negatives, and false negatives.
    records = []
    for metric, grp in detail_df.groupby("metric"):
        tp = int(((grp["has_hub"] == True)  & (grp["reject_H0"] == True)).sum())
        fp = int(((grp["has_hub"] == False) & (grp["reject_H0"] == True)).sum())
        tn = int(((grp["has_hub"] == False) & (grp["reject_H0"] == False)).sum())
        fn = int(((grp["has_hub"] == True)  & (grp["reject_H0"] == False)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")

        records.append({
            "metric":    metric,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "F1":        round(f1, 3),
        })

    summary_df = pd.DataFrame(records).sort_values("F1", ascending=False)
    return summary_df, detail_df


# ─────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    n_samples = 500
    model = "gnm"
    alpha = 0.05
    seed = 551

    summary, detail = evaluate(n_samples=n_samples, model=model, alpha=alpha, seed=seed)

    n_graphs = detail["graph_type"].nunique()
    n_positive = detail.loc[detail["has_hub"] == True, "graph_type"].nunique()
    n_negative = detail.loc[detail["has_hub"] == False, "graph_type"].nunique()
    print(
        f"Benchmark：graphs={n_graphs}, positives={n_positive}, negatives={n_negative}, "
        f"model={model}, n_samples={n_samples}, alpha={alpha}"
    )
    print()

    print(summary.to_string(index=False))

    print()
    print(detail.to_string(index=False))
