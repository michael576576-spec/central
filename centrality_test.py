"""
Statistical test for hub structure using random graph null models.

Workflow:
  1. Compute the spectral radius of the observed graph.
  2. Generate n_samples null graphs (hidden-variable model by default)
     and compute their spectral radii.
  3. Calculate a one-sided Monte Carlo p-value for hub structure.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from random_graph_models import null_model_from_graph


# ─────────────────────────────────────────────
# Metric calculation
# ─────────────────────────────────────────────

def compute_spectral_radius(G: nx.Graph) -> float:
    """
    Return the spectral radius λ₁ of the unweighted adjacency matrix.

    For an undirected nonnegative adjacency matrix, the largest algebraic
    eigenvalue is the spectral radius. Return NaN for an empty graph.
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
# Calculate graph metrics
# ─────────────────────────────────────────────

def compute_all_metrics(G: nx.Graph) -> dict[str, float]:
    """
    Calculate metrics for an observed or null graph.

    Parameters
    ----------
    G : Graph to evaluate.

    Returns
    -------
    A dict mapping metric names to floats, which may be NaN.
    """
    return {
        "spectral_radius": compute_spectral_radius(G),
    }


# ─────────────────────────────────────────────
# Null-model sampling
# ─────────────────────────────────────────────

def sample_null_metrics(
    G: nx.Graph,
    n_samples: int = 1000,
    model: str = "hidden",
    seed: int | None = None,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """
    Calculate observed metrics and sample their null distributions.

    Returns
    -------
    observed     : Metrics of the observed graph.
    null_records : Null samples for each metric, each with n_samples values.
    """
    observed = compute_all_metrics(G)

    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**31, size=n_samples).tolist()
    null_records: dict[str, list[float]] = {k: [] for k in observed}

    for i in range(n_samples):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G_null = null_model_from_graph(G, model=model, seed=child_seeds[i])
        null_vals = compute_all_metrics(G_null)
        for k, v in null_vals.items():
            if k in null_records:
                null_records[k].append(v)

    return observed, null_records


def empirical_p_value(observed: float, null_values: list[float] | np.ndarray) -> float:
    """One-sided Monte Carlo p-value with the observed statistic included."""
    values = np.asarray(null_values, dtype=float)
    valid = values[~np.isnan(values)]
    if len(valid) == 0 or np.isnan(observed):
        return float("nan")
    return float((np.sum(valid >= observed) + 1) / (len(valid) + 1))


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_null_distribution(
    G: nx.Graph,
    n_samples: int = 1000,
    model: str = "hidden",
    seed: int | None = None,
    metric: str = "spectral_radius",
    bins: int = 30,
    output_path: str | Path = "spectral_radius_null_distribution.png",
    show: bool = False,
) -> Path:
    """
    Plot a null-metric histogram with a line for the observed value.

    Parameters
    ----------
    G           : Observed undirected graph.
    n_samples   : Number of null-graph samples.
    model       : Null model: 'hidden', 'configuration-base',
                  'configuration-swap', 'gnm', or 'gnp'.
    seed        : Random seed.
    metric      : Metric name; defaults to 'spectral_radius'.
    bins        : Number of histogram bins.
    output_path : Path for the saved image.
    show        : Whether to display the plot interactively.

    Returns
    -------
    Path to the saved image.
    """
    import os
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "centrality_test_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    observed, null_records = sample_null_metrics(
        G,
        n_samples=n_samples,
        model=model,
        seed=seed,
    )
    if metric not in observed:
        raise ValueError(f"未知指标: {metric!r}，可选 {list(observed)}")

    obs_val = observed[metric]
    null_arr = np.array(null_records[metric], dtype=float)
    valid = null_arr[~np.isnan(null_arr)]
    p_val = empirical_p_value(obs_val, valid)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(valid, bins=bins, color="#4C78A8", alpha=0.75, edgecolor="white")
    if not np.isnan(obs_val):
        ax.axvline(
            obs_val,
            color="#D62728",
            linestyle="--",
            linewidth=2.2,
            label=f"Observed = {obs_val:.4f}, p = {p_val:.4f}",
        )
    ax.set_title(f"Null distribution of {metric} ({model}, n={n_samples})")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left")
    fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path



# ─────────────────────────────────────────────
# Main entry point: Monte Carlo test
# ─────────────────────────────────────────────

def test_hub(
    G: nx.Graph,
    n_samples: int = 1000,
    model: str = "hidden",
    alpha: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Test whether G has significant hub structure using spectral radius.

    Parameters
    ----------
    G         : Observed undirected graph (nx.Graph).
    n_samples : Number of null graphs to generate.
    model     : Null model: 'hidden' (default), 'configuration-base',
                'configuration-swap', 'gnm', or 'gnp'.
    alpha     : Significance level, default 0.05.
    seed      : Random seed; None leaves the sequence unseeded.

    Returns
    -------
    DataFrame with metric, observed, p_value, reject_H0, and conclusion.
    """
    observed, null_records = sample_null_metrics(
        G,
        n_samples=n_samples,
        model=model,
        seed=seed,
    )

    # A larger spectral radius supports hub structure: p = P(null >= observed).
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
                p_val = empirical_p_value(obs_val, valid)
                reject = p_val < alpha

        rows.append({
            "metric":     metric,
            "observed":   round(obs_val, 6) if not np.isnan(obs_val) else float("nan"),
            "p_value":    p_val,
            "reject_H0":  reject,
            "conclusion": "存在中心节点" if reject else "无显著中心节点",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    G = nx.karate_club_graph()
    print(f"原始图：N={G.number_of_nodes()}, M={G.number_of_edges()}")
    print()

    result = test_hub(G, n_samples=1000, model="hidden", alpha=0.05, seed=40)
    print(result.to_string(index=False))

    fig_path = plot_null_distribution(
        G,
        n_samples=1000,
        model="hidden",
        seed=40,
        output_path="spectral_radius_null_distribution.png",
    )
    print(f"\n零模型分布图已保存: {fig_path}")
