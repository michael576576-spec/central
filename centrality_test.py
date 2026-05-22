"""
基于随机图置换检验的中心节点统计检验

流程：
  1. 计算原始图的中心性指标（H、Freeman 中心势、变异系数）
  2. 用零模型（默认隐参数模型）生成 n_samples 个随机图并计算相同指标
  3. 单侧置换检验计算 p 值，判断原图是否存在显著中心节点
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import networkx as nx

from random_graph_models import null_model_from_graph


# ─────────────────────────────────────────────
# 打分函数（与指标计算解耦）
# ─────────────────────────────────────────────

def score_degree(G: nx.Graph) -> dict:
    """度中心性 C_i = k_i / (N-1)。"""
    return nx.degree_centrality(G)


def score_harmonic(G: nx.Graph) -> dict:
    """调和中心性 C_i = Σ_{j≠i} 1/d(i,j)，不连通节点对贡献 0。"""
    return nx.harmonic_centrality(G)


def score_coreness(G: nx.Graph) -> dict:
    """K-shell 分解值（coreness）。"""
    return nx.core_number(G)


# ─────────────────────────────────────────────
# 指标计算函数（接收打分字典，不感知打分方式）
# ─────────────────────────────────────────────

def compute_H(G: nx.Graph) -> float:
    """
    度异质性参数 H = <k²> / <k>²。
    <k> = 0 时返回 NaN。
    """
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    mean_k = degrees.mean()
    if mean_k == 0:
        return float("nan")
    return float((degrees ** 2).mean() / mean_k ** 2)


def compute_freeman(scores: dict, N: int, denominator: float) -> float:
    """
    通用弗里曼中心势 T = Σ(C_max - C_i) / denominator。

    Parameters
    ----------
    scores      : 节点打分字典 {node: score}
    N           : 节点数（用于一致性校验）
    denominator : 理论最大分母 D

    Returns
    -------
    float，分母 ≤ 0 或节点数 < 2 时返回 NaN
    """
    if denominator <= 0 or N < 2:
        return float("nan")
    values = np.array(list(scores.values()), dtype=float)
    c_max = values.max()
    return float((c_max - values).sum() / denominator)


def compute_cv(scores: dict) -> float:
    """
    变异系数 CV = σ / μ（σ 为样本标准差）。
    μ = 0 或样本数 < 2 时返回 NaN。
    """
    values = np.array(list(scores.values()), dtype=float)
    if len(values) < 2:
        return float("nan")
    mu = values.mean()
    if mu == 0:
        return float("nan")
    return float(values.std(ddof=1) / mu)


def compute_gini_degree(G: nx.Graph) -> float:
    """
    度序列的 Gini 系数。
    G = Σ(2i - n - 1) * v_i / (n * Σv_i)，v_i 为升序排列的度序列。
    所有度为 0 或节点数 < 2 时返回 NaN。
    """
    degrees = np.array(sorted(d for _, d in G.degree()), dtype=float)
    n = len(degrees)
    total = degrees.sum()
    if total == 0 or n < 2:
        return float("nan")
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float((weights * degrees).sum() / (n * total))


# ─────────────────────────────────────────────
# 弗里曼分母辅助函数
# ─────────────────────────────────────────────

def _denom_degree(N: int) -> float:
    """星形图下度中心性的理论最大分母 = N - 2。"""
    return float(N - 2)


def _denom_harmonic(N: int) -> float:
    """星形图下调和中心性的理论最大分母 = (N-1)(N-2)/2。"""
    return float((N - 1) * (N - 2) / 2)


def _denom_coreness(N: int, c_max: int) -> float:
    """Coreness 弗里曼分母 = c_max × (N-1)，固定 c_max 为原始图最大 coreness。"""
    return float(c_max * (N - 1))


# ─────────────────────────────────────────────
# 顶层组装：计算全部 7 个指标
# ─────────────────────────────────────────────

def compute_all_metrics(G: nx.Graph, c_max_ref: int | None = None) -> dict[str, float]:
    """
    计算原图或零模型图的全部 7 个指标。

    Parameters
    ----------
    G          : 待计算的图
    c_max_ref  : coreness 弗里曼分母中固定的最大 coreness，None 则取 G 本身的最大值

    Returns
    -------
    dict，键为指标名，值为 float（可能为 NaN）
    """
    N = G.number_of_nodes()
    scores_deg = score_degree(G)
    scores_har = score_harmonic(G)
    scores_cor = score_coreness(G)

    if c_max_ref is None:
        c_max_ref = max(scores_cor.values()) if scores_cor else 0

    return {
        "H":                compute_H(G),
        "freeman_degree":   compute_freeman(scores_deg, N, _denom_degree(N)),
        "freeman_harmonic": compute_freeman(scores_har, N, _denom_harmonic(N)),
        "freeman_coreness": compute_freeman(scores_cor, N, _denom_coreness(N, c_max_ref)),
        "cv_degree":        compute_cv(scores_deg),
        "cv_harmonic":      compute_cv(scores_har),
        "cv_coreness":      compute_cv(scores_cor),
        "gini_degree":      compute_gini_degree(G),
    }


# ─────────────────────────────────────────────
# 主入口：置换检验
# ─────────────────────────────────────────────

def test_hub(
    G: nx.Graph,
    n_samples: int = 1000,
    model: str = "hidden",
    alpha: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    基于随机图置换检验，判断图 G 中是否存在显著中心节点。

    Parameters
    ----------
    G         : 原始无向图（nx.Graph）
    n_samples : 生成随机零模型图的数量
    model     : 零模型类型，'hidden'（默认）、'configuration'、'gnm'、'gnp'
    alpha     : 显著性水平，默认 0.05
    seed      : 随机种子，None 则不固定

    Returns
    -------
    pd.DataFrame，列为 metric / observed / p_value / reject_H0 / conclusion
    """
    # 配置模型/隐参数模型保持度序列，基于度分布的指标无区分能力，跳过
    DEGREE_BASED = {"H", "freeman_degree", "cv_degree", "gini_degree"}
    skip = DEGREE_BASED if model in ("configuration", "hidden") else set()

    # 固定 coreness 分母为原始图的最大 coreness
    cores = nx.core_number(G)
    c_max_ref = max(cores.values()) if cores else 0

    # 计算原始图指标
    observed = {k: v for k, v in compute_all_metrics(G, c_max_ref).items() if k not in skip}

    # 生成可复现的子种子
    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**31, size=n_samples).tolist()

    # 收集零模型分布（只收集未跳过的指标）
    null_records: dict[str, list[float]] = {k: [] for k in observed}

    for i in range(n_samples):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G_null = null_model_from_graph(G, model=model, seed=child_seeds[i])
        null_vals = compute_all_metrics(G_null, c_max_ref)
        for k, v in null_vals.items():
            if k in null_records:
                null_records[k].append(v)

    # 计算单侧 p 值：p = P(null >= observed)
    rows = []
    for metric, obs_val in observed.items():
        null_arr = np.array(null_records[metric], dtype=float)

        if np.isnan(obs_val):
            p_val = float("nan")
            reject = False
        else:
            valid = null_arr[~np.isnan(null_arr)]
            if len(valid) == 0:
                p_val = float("nan")
                reject = False
            else:
                p_val = float(np.sum(valid >= obs_val) / len(valid))
                reject = p_val < alpha

        rows.append({
            "metric":     metric,
            "observed":   round(obs_val, 6) if not np.isnan(obs_val) else float("nan"),
            "p_value":    round(p_val, 4) if not np.isnan(p_val) else float("nan"),
            "reject_H0":  reject,
            "conclusion": "存在中心节点" if reject else "无显著中心节点",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 冒烟测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    G = nx.karate_club_graph()
    print(f"原始图：N={G.number_of_nodes()}, M={G.number_of_edges()}")
    print()

    result = test_hub(G, n_samples=1000, model="configuration", alpha=0.05, seed=42)
    print(result.to_string(index=False))
