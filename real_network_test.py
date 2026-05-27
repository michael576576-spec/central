"""
真实网络的中心节点检验

读取 network/ 目录下的边列表文件。每行前两列为一条无向边；
若存在第三列及后续列，按无权图口径忽略。
"""

from __future__ import annotations

from pathlib import Path
import time
import warnings

import networkx as nx
import pandas as pd

from centrality_test import test_hub


# ─────────────────────────────────────────────
# 网络加载
# ─────────────────────────────────────────────

NETWORK_DIR = Path(__file__).resolve().parent / "network"


def _from_edge_list(path: Path) -> nx.Graph:
    """读取边列表文件，返回简单无向无权图。"""
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
    返回待测网络列表，每项包含：
        name    : str
        G       : nx.Graph（无向无权简单图）
        path    : Path
        has_hub : bool（True = 认为存在中心节点）
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
# 单图检验
# ─────────────────────────────────────────────

def run_one(
    net: dict,
    n_samples: int,
    model: str,
    alpha: float,
    seed: int,
) -> list[dict]:
    """
    对单张真实网络运行 test_hub，返回每个指标的检验结果行。
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
# 汇总评估
# ─────────────────────────────────────────────

def evaluate(
    n_samples: int = 500,
    model: str = "hidden",
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对所有真实网络运行检验，计算每个指标的 precision / recall / F1。

    Returns
    -------
    summary_df : 每指标汇总（TP/FP/TN/FN/precision/recall/F1）
    detail_df  : 每张图 × 每指标的明细，含 elapsed_sec，按 N、M 从小到大排序
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
# 入口
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
