"""
基于随机图置换检验的中心节点统计检验

流程：
  1. 计算原始图的谱半径
  2. 用零模型（默认隐参数模型）生成 n_samples 个随机图并计算谱半径
  3. 单侧置换检验计算 p 值，判断原图是否存在显著中心节点
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import networkx as nx

from random_graph_models import null_model_from_graph


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────

def compute_spectral_radius(G: nx.Graph) -> float:
    """
    返回无权邻接矩阵谱半径 λ₁。

    对无向非负邻接矩阵，最大代数特征值即谱半径。空图返回 NaN。
    """
    import scipy.sparse.linalg as spla

    if G.number_of_nodes() == 0:
        return float("nan")
    if G.number_of_nodes() == 1:
        return 0.0

    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float, weight=None)
    try:
        vals = spla.eigsh(A, k=1, which="LA", return_eigenvectors=False)
        return float(vals[0])
    except Exception:
        vals = np.linalg.eigvalsh(A.toarray())
        return float(vals[-1])


# ─────────────────────────────────────────────
# 顶层组装：计算指标
# ─────────────────────────────────────────────

def compute_all_metrics(G: nx.Graph) -> dict[str, float]:
    """
    计算原图或零模型图的指标。

    Parameters
    ----------
    G : 待计算的图

    Returns
    -------
    dict，键为指标名，值为 float（可能为 NaN）
    """
    return {
        "spectral_radius": compute_spectral_radius(G),
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
    基于谱半径的随机图置换检验，判断图 G 中是否存在显著中心节点。

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
    # 计算原始图指标
    observed = compute_all_metrics(G)

    # 生成可复现的子种子
    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**31, size=n_samples).tolist()

    # 收集零模型分布（只收集未跳过的指标）
    null_records: dict[str, list[float]] = {k: [] for k in observed}

    for i in range(n_samples):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G_null = null_model_from_graph(G, model=model, seed=child_seeds[i])
        null_vals = compute_all_metrics(G_null)
        for k, v in null_vals.items():
            if k in null_records:
                null_records[k].append(v)

    # 谱半径越大越像有中心节点：p = P(null >= observed)
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

    result = test_hub(G, n_samples=1000, model="gnm", alpha=0.05, seed=40)
    print(result.to_string(index=False))
