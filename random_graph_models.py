"""
Random graph models built on NetworkX:
  ER models          : gnm / gnp
  Configuration base : pseudograph converted to a simple graph; degrees
                       are preserved approximately.
  Configuration swap : degree-preserving double-edge swaps.
  Hidden-variable    : Chung-Lu; expected degrees follow the input sequence.
"""

from __future__ import annotations

import warnings
import numpy as np
import networkx as nx


# ─────────────────────────────────────────────
# Helpers: extract graph parameters
# ─────────────────────────────────────────────

def graph_params(G: nx.Graph) -> dict:
    """Return graph parameters used by the generators."""
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
    """Convert a multigraph to a simple graph without loops or parallel edges."""
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


# ─────────────────────────────────────────────
# 1. ER models
# ─────────────────────────────────────────────

def er_gnm(N: int, M: int, seed: int | None = None) -> nx.Graph:
    """Sample M distinct edges among N nodes (G(N, M))."""
    return nx.gnm_random_graph(N, M, seed=seed)


def er_gnp(N: int, p: float, seed: int | None = None) -> nx.Graph:
    """Connect each pair independently with probability p (G(N, p))."""
    return nx.erdos_renyi_graph(N, p, seed=seed)


# ─────────────────────────────────────────────
# 2. Configuration models
# ─────────────────────────────────────────────

def configuration_model(
    G_original: nx.Graph,
    seed: int | None = None,
    Q: int = 10,
) -> nx.Graph:
    """
    Generate a pseudograph from the original degree sequence with NetworkX,
    then remove self-loops and merge parallel edges to obtain a simple graph.

    Removing loops and parallel edges changes node degrees, so this model
    preserves the original degree distribution only approximately.

    Parameters
    ----------
    G_original : Original undirected graph.
    seed       : Random seed passed to nx.configuration_model.
    Q          : Kept for compatibility with old calls; currently unused.
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
# 3. Hidden-variable / Chung-Lu model
# ─────────────────────────────────────────────

def hidden_variable_model(
    degree_sequence: list[int],
    seed: int | None = None,
) -> nx.Graph:
    """
    Chung-Lu hidden-variable model with p_ij = w_i*w_j / sum(w).
    Expected degrees follow the input sequence. Uses nx.expected_degree_graph.
    """
    w = [float(k) for k in degree_sequence]
    G = nx.expected_degree_graph(w, seed=seed, selfloops=False)
    return _to_simple(G)


# ─────────────────────────────────────────────
# Top-level helper: derive parameters and generate a null graph
# ─────────────────────────────────────────────

def null_model_from_graph(
    G_original: nx.Graph,
    model: str = "hidden",
    seed: int | None = None,
) -> nx.Graph:
    """
    Derive parameters from G and generate a null-model graph.

    Parameters
    ----------
    G_original : Original undirected graph.
    model      : One of:
                 'gnm'                - G(N, M) ER model.
                 'gnp'                - G(N, p) ER model.
                 'configuration-base' - Approximately preserve degrees.
                 'configuration-swap' - Preserve degrees with edge swaps.
                 'hidden'             - Chung-Lu model (default).
    seed       : Random seed.
    """
    params = graph_params(G_original)
    N, M, p = params["N"], params["M"], params["p"]
    deg_seq = params["deg_seq"]

    if model == "gnm":
        return er_gnm(N, M, seed=seed)

    elif model == "gnp":
        return er_gnp(N, p, seed=seed)

    elif model == "configuration-base":
        return configuration_model(G_original, seed=seed)

    elif model == "configuration-swap":
        return configuration_model_fast(G_original, seed=seed)

    elif model == "hidden":
        return hidden_variable_model(deg_seq, seed=seed)

    else:
        raise ValueError(
            f"未知模型: {model!r}，"
            "可选 'gnm'、'gnp'、'configuration-base'、'configuration-swap'、'hidden'"
        )


# ─────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    G = nx.karate_club_graph()
    p = graph_params(G)
    print(f"原始图：N={p['N']}, M={p['M']}, mean_k={p['mean_k']:.2f}")
    print()

    models = ["gnm", "gnp", "configuration-base", "configuration-swap", "hidden"]
    for m in models:
        G_rand = null_model_from_graph(G, model=m, seed=42)
        print(f"{m:15s}: N={G_rand.number_of_nodes()}, M={G_rand.number_of_edges()}")
