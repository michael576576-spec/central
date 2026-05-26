"""
中心节点检测方法的检测能力评估

用已知答案的合成图（正例 = 有中心节点，负例 = 无中心节点）评估
centrality_test.test_hub 中每个指标的 precision / recall / F1。
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import networkx as nx

from centrality_test import test_hub


# ─────────────────────────────────────────────
# 合成图构造
# ─────────────────────────────────────────────

def _hub_er(n_total: int, hub_degree_ratio: float, seed: int) -> nx.Graph:
    """
    hub + ER 背景噪声图。
    hub 节点 0 连接 floor(n_total * hub_degree_ratio) 个随机节点，
    其余节点之间按 p = 2*M_background / (N*(N-1)) 随机连边。
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_total))

    n_hub_edges = max(1, int(n_total * hub_degree_ratio))
    hub_neighbors = rng.choice(range(1, n_total), size=min(n_hub_edges, n_total - 1), replace=False)
    for v in hub_neighbors:
        G.add_edge(0, v)

    # 背景边：密度约等于 hub 边密度的一半
    p_bg = n_hub_edges / (n_total * (n_total - 1) / 2) * 0.5
    for u in range(1, n_total):
        for v in range(u + 1, n_total):
            if rng.random() < p_bg:
                G.add_edge(u, v)
    return G


def _double_hub(n_total: int, hub_degree: int, seed: int) -> nx.Graph:
    """
    两个 hub 节点（0, 1）各连接 hub_degree 个不同叶子节点。
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


def make_benchmark_graphs(seed: int = 42) -> list[dict]:
    """
    构造一批已知答案的合成图。

    Returns
    -------
    list of dict，每项包含：
        G           : nx.Graph
        has_hub     : bool（True = 正例，False = 负例）
        graph_type  : str（描述性标签）
        N           : int
    """
    rng = np.random.default_rng(seed)
    entries = []

    def child(i):
        return int(rng.integers(0, 2**31))

    # ── 正例：有中心节点 ──────────────────────────
    # 纯星形图（极端 hub）
    for n_leaves in [19, 49, 99]:
        entries.append({
            "G": nx.star_graph(n_leaves),
            "has_hub": True,
            "graph_type": f"star_N{n_leaves + 1}",
            "N": n_leaves + 1,
        })

    # hub + ER 背景噪声
    for ratio in [0.3, 0.5, 0.8]:
        G = _hub_er(50, ratio, seed=child(0))
        entries.append({
            "G": G,
            "has_hub": True,
            "graph_type": f"hub_er_ratio{ratio}",
            "N": 50,
        })

    # 双 hub 图
    G = _double_hub(60, hub_degree=20, seed=child(1))
    entries.append({
        "G": G,
        "has_hub": True,
        "graph_type": "double_hub_N60",
        "N": 60,
    })

    # ── 负例：无中心节点 ──────────────────────────
    # ER 随机图（均匀随机）
    for N, p in [(50, 0.08), (100, 0.06)]:
        G = nx.erdos_renyi_graph(N, p, seed=child(2))
        entries.append({
            "G": G,
            "has_hub": False,
            "graph_type": f"er_N{N}_p{p}",
            "N": N,
        })

    # 规则图（所有节点度相同，最强负例）
    for N, d in [(50, 4), (100, 6)]:
        G = nx.random_regular_graph(d, N, seed=child(3))
        entries.append({
            "G": G,
            "has_hub": False,
            "graph_type": f"regular_N{N}_d{d}",
            "N": N,
        })

    # BA 无标度图（有轻微幂律但无极端 hub，边界情况）
    for N, m in [(50, 2), (100, 3)]:
        G = nx.barabasi_albert_graph(N, m, seed=child(4))
        entries.append({
            "G": G,
            "has_hub": False,
            "graph_type": f"ba_N{N}_m{m}",
            "N": N,
        })

    return entries


# ─────────────────────────────────────────────
# 单图检验
# ─────────────────────────────────────────────

def run_one(
    entry: dict,
    n_samples: int,
    model: str,
    alpha: float,
    seed: int,
) -> list[dict]:
    """
    对单张图运行 test_hub，返回每个指标的检验结果行。
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
            "has_hub":    entry["has_hub"],
            "metric":     row["metric"],
            "observed":   row["observed"],
            "p_value":    row["p_value"],
            "reject_H0":  row["reject_H0"],
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对所有合成图运行检验，计算每个指标的 precision / recall / F1。

    Returns
    -------
    summary_df : 每指标汇总（TP/FP/TN/FN/precision/recall/F1）
    detail_df  : 每张图 × 每指标的明细
    """
    graphs = make_benchmark_graphs(seed=seed)
    rng = np.random.default_rng(seed + 1)
    child_seeds = rng.integers(0, 2**31, size=len(graphs)).tolist()

    all_rows = []
    for i, entry in enumerate(graphs):
        print(f"  [{i+1}/{len(graphs)}] {entry['graph_type']} (has_hub={entry['has_hub']})")
        rows = run_one(entry, n_samples=n_samples, model=model, alpha=alpha,
                       seed=child_seeds[i])
        all_rows.extend(rows)

    detail_df = pd.DataFrame(all_rows)

    # 计算 TP/FP/TN/FN
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
    print("=== 构造合成图并运行检验 ===\n")
    summary, detail = evaluate(n_samples=500, model="gnm", alpha=0.05, seed=42)

    print("\n=== 每指标检测能力汇总 ===")
    print(summary.to_string(index=False))

    print("\n=== 每张图明细（pivot） ===")
    pivot = detail.pivot_table(
        index=["graph_type", "has_hub"],
        columns="metric",
        values="reject_H0",
    ).astype(object)
    print(pivot.to_string())
