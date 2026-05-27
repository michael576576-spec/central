"""
真实网络的中心节点检验

网络来源：
  - NetworkX 内置：karate_club、les_miserables、florentine_families
  - 本地 mat 文件：USAir（美国航空）、Celegans（线虫神经网络）
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import networkx as nx
import scipy.io as sio

from centrality_test import test_hub


# ─────────────────────────────────────────────
# 网络加载
# ─────────────────────────────────────────────

MAT_DIR = "/Users/wql/Downloads/research/Interpretable-Link-Prediction/dataset"


def _from_mat(filename: str) -> nx.Graph:
    """加载 .mat 邻接矩阵并转为简单无向图。"""
    data = sio.loadmat(f"{MAT_DIR}/{filename}")
    key = [k for k in data if not k.startswith("_")][0]
    A = data[key]
    G = nx.from_scipy_sparse_array(A) if hasattr(A, "toarray") else nx.from_numpy_array(np.array(A))
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _unweight(G: nx.Graph) -> nx.Graph:
    """去除边权重，返回简单无向无权图。"""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    return H


def load_networks() -> list[dict]:
    """
    返回待测网络列表，每项包含：
        name        : str
        G           : nx.Graph（无向无权简单图）
        description : str（网络背景说明）
    """
    nets = []

    # ── NetworkX 内置 ──────────────────────────
    nets.append({
        "name": "karate_club",
        "G": _unweight(nx.karate_club_graph()),
        "description": "空手道俱乐部社交网络（Zachary 1977），34人，因派系分裂而著名",
    })
    nets.append({
        "name": "les_miserables",
        "G": _unweight(nx.les_miserables_graph()),
        "description": "悲惨世界人物共现网络，77个角色，冉阿让为核心人物",
    })
    nets.append({
        "name": "florentine_families",
        "G": _unweight(nx.florentine_families_graph()),
        "description": "佛罗伦萨家族婚姻网络（Padgett 1994），15个家族，美第奇家族为核心",
    })

    # ── 本地 mat 文件 ──────────────────────────
    nets.append({
        "name": "USAir",
        "G": _from_mat("USAir.mat"),
        "description": "美国航空网络，332个机场，枢纽机场（如芝加哥、亚特兰大）度极高",
    })
    nets.append({
        "name": "Celegans",
        "G": _from_mat("Celegans.mat"),
        "description": "线虫（C. elegans）神经网络，297个神经元，度分布有异质性",
    })

    return nets


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main(n_samples: int = 500, model: str = "hidden", alpha: float = 0.05, seed: int = 42):
    networks = load_networks()

    all_rows = []
    for net in networks:
        name = net["name"]
        G = net["G"]
        desc = net["description"]
        N, M = G.number_of_nodes(), G.number_of_edges()
        degs = sorted([d for _, d in G.degree()], reverse=True)
        print(f"\n{'='*60}")
        print(f"[{name}] N={N}, M={M}, top3_deg={degs[:3]}, mean_deg={np.mean(degs):.2f}")
        print(f"  {desc}")
        print(f"  零模型: {model}, 样本数: {n_samples}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = test_hub(G, n_samples=n_samples, model=model, alpha=alpha, seed=seed)

        print(result.to_string(index=False))

        for _, row in result.iterrows():
            all_rows.append({"network": name, **row.to_dict()})

    # ── 汇总：各网络被多少指标判定为有中心 ──
    # print(f"\n{'='*60}")
    # print("汇总：各网络被判定为「存在中心节点」的指标数")
    # summary_df = (
    #     pd.DataFrame(all_rows)
    #     .groupby("network")["reject_H0"]
    #     .agg(total="count", n_reject="sum")
    #     .assign(ratio=lambda df: (df["n_reject"] / df["total"]).map("{:.0%}".format))
    #     .reset_index()
    # )
    # print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main(n_samples=500, model="gnm", alpha=0.05, seed=42)
