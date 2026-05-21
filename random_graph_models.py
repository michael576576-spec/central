"""
随机图模型实现
包含三种模型：
1. ER 模型（Erdős–Rényi）：完全随机化
2. 配置模型（Configuration Model）：度保持随机化，节点度序列严格相等
3. 隐参数模型（Hidden Variable Model）：度序列期望相等
"""

from __future__ import annotations

import random
import itertools
import networkx as nx
import numpy as np


# ─────────────────────────────────────────────
# 1. ER 模型
# ─────────────────────────────────────────────

def er_gnm(N: int, M: int, seed: int | None = None) -> nx.Graph:
    """
    G(N, M) 模型：从 N 个节点的所有可能边中随机选取 M 条边。

    Parameters
    ----------
    N : 节点数
    M : 边数
    seed : 随机种子

    Returns
    -------
    nx.Graph
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    all_edges = list(itertools.combinations(range(N), 2))
    if M > len(all_edges):
        raise ValueError(f"M={M} 超过最大可能边数 {len(all_edges)}")

    chosen = rng.sample(all_edges, M)
    G.add_edges_from(chosen)
    return G


def er_gnp(N: int, p: float, seed: int | None = None) -> nx.Graph:
    """
    G(N, p) 模型：每对节点以概率 p 独立连边。

    Parameters
    ----------
    N : 节点数
    p : 连边概率，取值 [0, 1]
    seed : 随机种子

    Returns
    -------
    nx.Graph
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for u, v in itertools.combinations(range(N), 2):
        if rng.random() < p:
            G.add_edge(u, v)
    return G


# ─────────────────────────────────────────────
# 2. 配置模型（度序列严格相等）
# ─────────────────────────────────────────────

def configuration_model(
    G_original: nx.Graph,
    seed: int | None = None,
    Q: int = 10,
) -> nx.Graph:
    """
    配置模型：以原始图为起点，通过 Markov 链双边互换（double edge swap）
    对边进行充分随机化，节点度序列严格保持不变。

    算法（MCMC edge-swap）：
    每次随机选取两条边 (u,v) 和 (s,t)（4 个端点互不相同），
    尝试重连为 (u,s),(v,t) 或 (u,t),(v,s)，
    若不产生重边则接受交换。重复 Q*M 次（Q 默认 10）。

    Parameters
    ----------
    G_original : 原始无向简单图，用于初始化并提取度序列
    seed : 随机种子
    Q : 每条边平均互换次数（越大随机化越充分，默认 10）

    Returns
    -------
    nx.Graph
        度序列与原图严格相同的随机简单图
    """
    rng = random.Random(seed)

    G = G_original.copy()
    edges = list(G.edges())
    M = len(edges)
    n_swaps = Q * M

    swapped = 0
    attempts = 0
    max_attempts = 100 * n_swaps

    while swapped < n_swaps and attempts < max_attempts:
        attempts += 1
        edges = list(G.edges())
        e1, e2 = rng.sample(edges, 2)
        u, v = e1
        s, t = e2

        # 4 个端点必须互不相同
        if len({u, v, s, t}) < 4:
            continue

        # 随机选一种重连方式
        if rng.random() < 0.5:
            new1, new2 = (u, s), (v, t)
        else:
            new1, new2 = (u, t), (v, s)

        # 检查新边是否已存在（避免重边）
        if G.has_edge(*new1) or G.has_edge(*new2):
            continue

        G.remove_edge(u, v)
        G.remove_edge(s, t)
        G.add_edge(*new1)
        G.add_edge(*new2)
        swapped += 1

    return G


# ─────────────────────────────────────────────
# 3. 隐参数模型（度序列期望相等）
# ─────────────────────────────────────────────

def hidden_variable_model(
    degree_sequence: list[int],
    seed: int | None = None,
) -> nx.Graph:
    """
    隐参数模型（Chung–Lu 模型）：
    每个节点 i 的隐参数为其度数 θ_i = k_i，
    两节点 i, j 之间连边的概率为：

        p_ij = θ_i * θ_j / (∑_v θ_v) = k_i * k_j / (2M)

    该模型保证节点度数的**期望**等于原始度序列，但实际度数是随机变量。

    Parameters
    ----------
    degree_sequence : 原始网络的度序列，长度为 N
    seed : 随机种子

    Returns
    -------
    nx.Graph
    """
    rng = random.Random(seed)

    N = len(degree_sequence)
    k = np.array(degree_sequence, dtype=float)
    two_M = k.sum()  # ∑_v k_v = 2M

    if two_M == 0:
        G = nx.Graph()
        G.add_nodes_from(range(N))
        return G

    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i + 1, N):
            p_ij = min(k[i] * k[j] / two_M, 1.0)  # 概率截断至 1
            if rng.random() < p_ij:
                G.add_edge(i, j)

    return G


# ─────────────────────────────────────────────
# 工具函数：从已有图中提取参数并生成零模型
# ─────────────────────────────────────────────

def null_model_from_graph(
    G_original: nx.Graph,
    model: str = "hidden",
    seed: int | None = None,
) -> nx.Graph:
    """
    根据原始图 G 自动提取参数，生成对应的零模型随机图。

    Parameters
    ----------
    G_original : 原始无向图
    model : 模型类型，可选 'gnm'、'gnp'、'configuration'、'hidden'
    seed : 随机种子

    Returns
    -------
    nx.Graph
    """
    N = G_original.number_of_nodes()
    M = G_original.number_of_edges()
    nodes = list(G_original.nodes())
    deg_seq = [G_original.degree(v) for v in nodes]

    if model == "gnm":
        return er_gnm(N, M, seed=seed)

    elif model == "gnp":
        max_edges = N * (N - 1) / 2
        p = M / max_edges if max_edges > 0 else 0.0
        return er_gnp(N, p, seed=seed)

    elif model == "configuration":
        return configuration_model(G_original, seed=seed)

    elif model == "hidden":
        return hidden_variable_model(deg_seq, seed=seed)

    else:
        raise ValueError(f"未知模型类型: {model!r}，请选择 'gnm'、'gnp'、'configuration' 或 'hidden'")


# ─────────────────────────────────────────────
# 示例演示
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 构造一个示例原始图（Karate Club 网络）
    G = nx.karate_club_graph()
    N = G.number_of_nodes()
    M = G.number_of_edges()
    deg_seq = [d for _, d in G.degree()]

    print(f"原始图：N={N}, M={M}")
    print(f"度序列均值：{np.mean(deg_seq):.3f}, 最大度：{max(deg_seq)}")
    print()

    # ER G(N,M) 模型
    g_gnm = er_gnm(N, M, seed=42)
    print(f"G(N,M)    模型：N={g_gnm.number_of_nodes()}, M={g_gnm.number_of_edges()}")

    # ER G(N,p) 模型
    g_gnp = er_gnp(N, p=2 * M / (N * (N - 1)), seed=42)
    print(f"G(N,p)    模型：N={g_gnp.number_of_nodes()}, M={g_gnp.number_of_edges()}")

    # 配置模型（传入原始图，edge-swap 保持度序列）
    g_conf = configuration_model(G, seed=42)
    conf_deg = sorted([d for _, d in g_conf.degree()])
    print(f"配置模型  ：N={g_conf.number_of_nodes()}, M={g_conf.number_of_edges()}, "
          f"度序列匹配={conf_deg == sorted(deg_seq)}")

    # 隐参数模型
    g_hidden = hidden_variable_model(deg_seq, seed=42)
    hidden_deg = [d for _, d in g_hidden.degree()]
    print(f"隐参数模型：N={g_hidden.number_of_nodes()}, M={g_hidden.number_of_edges()}, "
          f"度序列均值={np.mean(hidden_deg):.3f}")
