"""
随机图模型实现
基于 NetworkX 内置函数，提供以下模型：
  ER 模型   : gnm / gnp
  配置模型  : configuration（伪图配置模型转简单图，近似保留度分布）
  隐参数模型: hidden（Chung-Lu，度序列期望保持）
"""

from __future__ import annotations

import warnings
import numpy as np
import networkx as nx


# ─────────────────────────────────────────────
# 辅助：从图中提取常用参数
# ─────────────────────────────────────────────

def graph_params(G: nx.Graph) -> dict:
    """返回图的基本参数字典，供各生成函数使用。"""
    N = G.number_of_nodes()
    M = G.number_of_edges()
    max_edges = N * (N - 1) / 2
    deg_seq = [d for _, d in G.degree()]
    return {
        "N": N,
        "M": M,
        "p": M / max_edges if max_edges > 0 else 0.0,
        "deg_seq": deg_seq,
        "mean_k": float(np.mean(deg_seq)) if deg_seq else 0.0,
    }


def _to_simple(G: nx.Graph) -> nx.Graph:
    """多重图转简单图（去自环和重边），返回新 Graph。"""
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


# ─────────────────────────────────────────────
# 1. ER 模型
# ─────────────────────────────────────────────

def er_gnm(N: int, M: int, seed: int | None = None) -> nx.Graph:
    """G(N, M)：从 N 个节点中随机选取 M 条不重复边。"""
    return nx.gnm_random_graph(N, M, seed=seed)


def er_gnp(N: int, p: float, seed: int | None = None) -> nx.Graph:
    """G(N, p)：每对节点以概率 p 独立连边。"""
    return nx.erdos_renyi_graph(N, p, seed=seed)


# ─────────────────────────────────────────────
# 2. 配置模型（伪图转简单图，近似保留度分布）
# ─────────────────────────────────────────────

def configuration_model(
    G_original: nx.Graph,
    seed: int | None = None,
    Q: int = 10,
) -> nx.Graph:
    """
    配置模型：基于原始图的度序列，使用 NetworkX configuration_model
    生成允许自环和重边的伪图，然后转成简单图并去除自环。

    这个版本速度较快，但转简单图会合并重边、删除自环，因此节点度
    不再保证与原图严格相同，只是近似保留原始度分布。

    Parameters
    ----------
    G_original : 原始无向图
    seed       : 随机种子（传给 nx.configuration_model）
    Q          : 保留该参数以兼容旧调用；当前实现不使用
    """
    nodes = list(G_original.nodes())
    deg_seq = [G_original.degree(n) for n in nodes]

    G_multi = nx.configuration_model(deg_seq, seed=seed)
    G_simple = _to_simple(G_multi)

    mapping = dict(enumerate(nodes))
    return nx.relabel_nodes(G_simple, mapping)


def configuration_model_fast(
    G_original: nx.Graph,
    seed: int | None = None,
    Q: int = 3,
    max_tries_per_swap: int = 5,
) -> nx.Graph:
    """
    Faster exact-degree simple-graph configuration model.

    This keeps the same semantics as configuration_model: start from the
    original simple graph and randomize it with degree-preserving double-edge
    swaps. The difference is a smaller default swap budget and a tighter
    max_tries cap, which avoids spending a long time on degree sequences with
    very few legal swaps.
    """
    G = G_original.copy()
    M = G.number_of_edges()
    if M < 2:
        return G

    nswap = max(1, Q * M)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nx.double_edge_swap(
                G,
                nswap=nswap,
                max_tries=nswap * max_tries_per_swap,
                seed=seed,
            )
    except nx.NetworkXAlgorithmError:
        pass
    return G


# ─────────────────────────────────────────────
# 3. 隐参数模型 / Chung-Lu（度序列期望相等）
# ─────────────────────────────────────────────

def hidden_variable_model(
    degree_sequence: list[int],
    seed: int | None = None,
) -> nx.Graph:
    """
    Chung-Lu 隐参数模型：节点 i,j 连边概率 p_ij = w_i*w_j / Σw，
    保证度序列期望等于输入序列。使用 nx.expected_degree_graph 实现。
    """
    w = [float(k) for k in degree_sequence]
    G = nx.expected_degree_graph(w, seed=seed, selfloops=False)
    return _to_simple(G)


# ─────────────────────────────────────────────
# 顶层工具：从已有图提取参数并生成零模型
# ─────────────────────────────────────────────

def null_model_from_graph(
    G_original: nx.Graph,
    model: str = "hidden",
    seed: int | None = None,
) -> nx.Graph:
    """
    根据原始图 G 自动提取参数，生成对应零模型随机图。

    Parameters
    ----------
    G_original : 原始无向图
    model      : 模型类型，可选：
                   'gnm'           — G(N,M) ER 模型
                   'gnp'           — G(N,p) ER 模型
                   'configuration' — 配置模型（近似保留度分布）
                   'hidden'        — Chung-Lu 隐参数模型（默认）
    seed       : 随机种子
    """
    params = graph_params(G_original)
    N, M, p = params["N"], params["M"], params["p"]
    deg_seq = params["deg_seq"]

    if model == "gnm":
        return er_gnm(N, M, seed=seed)

    elif model == "gnp":
        return er_gnp(N, p, seed=seed)

    elif model == "configuration":
        return configuration_model(G_original, seed=seed)

    elif model == "configuration_fast":
        return configuration_model_fast(G_original, seed=seed)

    elif model == "hidden":
        return hidden_variable_model(deg_seq, seed=seed)

    else:
        raise ValueError(
            f"未知模型: {model!r}，"
            "可选 'gnm'、'gnp'、'configuration'、'configuration_fast'、'hidden'"
        )


# ─────────────────────────────────────────────
# 示例演示
# ─────────────────────────────────────────────

if __name__ == "__main__":
    G = nx.karate_club_graph()
    p = graph_params(G)
    print(f"原始图：N={p['N']}, M={p['M']}, mean_k={p['mean_k']:.2f}")
    print()

    models = ["gnm", "gnp", "configuration", "configuration_fast", "hidden"]
    for m in models:
        G_rand = null_model_from_graph(G, model=m, seed=42)
        print(f"{m:15s}: N={G_rand.number_of_nodes()}, M={G_rand.number_of_edges()}")
