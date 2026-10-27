"""
Hub-structure tests on real networks.

Read edge-list files from network/. The first two columns of each line
define an undirected edge; any remaining columns are ignored.
"""

from __future__ import annotations

from pathlib import Path
import time
import warnings

import networkx as nx
import pandas as pd

from centrality_test import test_hub


# ─────────────────────────────────────────────
# Load networks
# ─────────────────────────────────────────────

NETWORK_DIR = Path(__file__).resolve().parent / "network"


def _from_edge_list(path: Path) -> nx.Graph:
    """Read an edge list as a simple, undirected, unweighted graph."""
    G = nx.Graph()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            G.add_edge(parts[0], parts[1])

    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def load_networks(network_dir: str | Path = NETWORK_DIR) -> list[dict]:
    """
    Return network records, each containing:
        name    : str
        G       : nx.Graph (simple, undirected, unweighted)
        path    : Path
        has_hub : bool (True = treated as having hub structure)
    """
    network_dir = Path(network_dir)
    nets = []
    for path in sorted(network_dir.glob("*.txt")):
        nets.append({
            "name": path.stem,
            "G": _from_edge_list(path),
            "path": path,
            "has_hub": True,
        })
    return nets


# ─────────────────────────────────────────────
# Test one graph
# ─────────────────────────────────────────────

def run_one(
    net: dict,
    n_samples: int,
    model: str,
    alpha: float,
    seed: int,
) -> list[dict]:
    """
    Run test_hub on one real network and return a row per metric.
    """
    G = net["G"]
    started_at = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_df = test_hub(G, n_samples=n_samples, model=model,
                             alpha=alpha, seed=seed)
    elapsed_sec = time.perf_counter() - started_at

    rows = []
    for _, row in result_df.iterrows():
        rows.append({
            "graph_type": net["name"],
            "N":          G.number_of_nodes(),
            "M":          G.number_of_edges(),
            "has_hub":    net["has_hub"],
            "metric":     row["metric"],
            "observed":   row["observed"],
            "p_value":    row["p_value"],
            "reject_H0":  row["reject_H0"],
            "conclusion": row["conclusion"],
            "elapsed_sec": round(elapsed_sec, 4),
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
    Test all real networks and calculate precision, recall, and F1 per metric.

    Returns
    -------
    summary_df : Per-metric TP, FP, TN, FN, precision, recall, and F1.
    detail_df  : Per-graph, per-metric results including elapsed_sec,
                 sorted by N and M.
    """
    networks = load_networks()

    all_rows = []
    for i, net in enumerate(networks):
        if verbose:
            print(f"  [{i+1}/{len(networks)}] {net['name']} (has_hub={net['has_hub']})")
        rows = run_one(net, n_samples=n_samples, model=model, alpha=alpha, seed=seed)
        all_rows.extend(rows)

    detail_df = (
        pd.DataFrame(all_rows)
        .sort_values(["N", "M", "graph_type"])
        .reset_index(drop=True)
    )

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
    n_samples = 1000
    model = "gnm"
    alpha = 0.05
    seed = 42

    summary, detail = evaluate(n_samples=n_samples, model=model, alpha=alpha, seed=seed)

    print(summary.to_string(index=False))

    print()
    print(detail.to_string(index=False))
