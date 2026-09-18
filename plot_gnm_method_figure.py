"""Create a publication-style overview of the GNM spectral-radius test."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "centrality_test_mpl_cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import networkx as nx
import numpy as np

from centrality_test import sample_null_metrics
from generate_gnm_components import (
    connected_gnm_graph,
    make_positive_graph,
    p_value,
    select_mid_null_graph,
)


def draw_graph(ax: plt.Axes, graph: nx.Graph, seed: int, highlight_hub: bool = False) -> None:
    pos = nx.spring_layout(graph, seed=seed, k=0.75, iterations=150)
    degree = np.asarray([graph.degree[node] for node in graph], dtype=float)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#AAB4C0", width=0.75, alpha=0.8)
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=degree, cmap="viridis",
        node_size=22 + 10 * degree, linewidths=0.5, edgecolors="white",
    )
    if highlight_hub:
        hub = max(graph.nodes, key=graph.degree)
        nx.draw_networkx_nodes(
            graph, pos, nodelist=[hub], ax=ax, node_color="#F2C94C",
            node_size=105, edgecolors="#D62728", linewidths=2.0,
        )
    ax.set_axis_off()


def draw_histogram(ax: plt.Axes, observed: float, null_values: list[float], p_val: float) -> None:
    values = np.asarray(null_values, dtype=float)
    values = values[~np.isnan(values)]
    ax.hist(values, bins=20, color="#B8B8B8", edgecolor="white", linewidth=0.5)
    xmin, xmax = ax.get_xlim()
    if observed > xmax:
        ax.set_xlim(xmin, observed + 0.08 * (observed - xmin))
    elif observed < xmin:
        ax.set_xlim(observed - 0.08 * (xmax - observed), xmax)
    ymax = ax.get_ylim()[1]
    ax.annotate(
        rf"$\rho$ = {observed:.2f}", xy=(observed, ymax * 0.08), xycoords="data",
        xytext=(0.72, 0.82), textcoords="axes fraction", color="#D62728",
        ha="left", va="center", fontsize=11,
        arrowprops={"arrowstyle": "-|>", "color": "#D62728", "lw": 1.6},
    )
    p_label = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(
        0.03,
        0.93,
        p_label,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.5},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def add_arrow(
    fig: plt.Figure,
    x_left: float,
    x_right: float,
    y: float,
    half_length: float,
) -> None:
    """Draw a long horizontal process arrow in the whitespace between columns."""
    center = (x_left + x_right) / 2
    fig.add_artist(
        plt.Line2D(
            [center - half_length, center + half_length],
            [y, y],
            transform=fig.transFigure,
            color="#506176",
            linewidth=2.0,
            solid_capstyle="round",
            marker=">",
            markevery=[1],
            markersize=8,
        )
    )


def build_figure(output: Path, n_samples: int = 500, seed: int = 20260919) -> Path:
    positive = make_positive_graph()
    negative = select_mid_null_graph(30, 50, n_samples, seed + 100)
    examples = [(positive, True), (negative, False)]

    fig = plt.figure(figsize=(14, 8.5), dpi=180)
    outer = GridSpec(2, 3, figure=fig, width_ratios=[1.08, 1.15, 1.18], wspace=0.30, hspace=0.40)
    left_axes: list[plt.Axes] = []
    middle_axes: list[list[plt.Axes]] = []
    right_axes: list[plt.Axes] = []
    histogram_values: list[float] = []

    for row, (graph, highlight) in enumerate(examples):
        observed, null_records = sample_null_metrics(graph, n_samples=n_samples, model="gnm", seed=seed + row)
        observed_value = float(observed["spectral_radius"])
        null_values = null_records["spectral_radius"]
        p_val = p_value(observed_value, null_values)
        histogram_values.extend([observed_value, *null_values])

        left = fig.add_subplot(outer[row, 0])
        draw_graph(left, graph, seed=12, highlight_hub=False)
        left.text(
            0.02,
            0.94,
            rf"$\rho$ = {observed_value:.2f}",
            transform=left.transAxes,
            fontsize=11,
            color="#303846",
        )
        left_axes.append(left)

        middle_spec = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[row, 1], wspace=0.08, hspace=0.08)
        middle_group: list[plt.Axes] = []
        rng = np.random.default_rng(seed + 300 + row)
        for index in range(4):
            null_graph = connected_gnm_graph(
                graph.number_of_nodes(), graph.number_of_edges(), seed=int(rng.integers(2**31))
            )
            mini = fig.add_subplot(middle_spec[index // 2, index % 2])
            draw_graph(mini, null_graph, seed=index + 30)
            middle_group.append(mini)
        middle_axes.append(middle_group)

        right = fig.add_subplot(outer[row, 2])
        draw_histogram(right, observed_value, null_values, p_val)
        right_axes.append(right)

    # Use one shared x-range and tick grid for both null distributions so the
    # two rows are directly comparable.  The range includes both simulated
    # values and observed statistics, with a small visual margin.
    finite_values = np.asarray(histogram_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    x_min = float(np.min(finite_values))
    x_max = float(np.max(finite_values))
    x_margin = max(0.06, 0.05 * (x_max - x_min))
    shared_min = x_min - x_margin
    shared_max = x_max + x_margin
    tick_start = np.floor(shared_min * 2) / 2
    tick_stop = np.ceil(shared_max * 2) / 2
    shared_ticks = np.arange(tick_start, tick_stop + 0.25, 0.5)
    for right in right_axes:
        right.set_xlim(shared_min, shared_max)
        right.set_xticks(shared_ticks)

    # Shift the comparison panels slightly right, leaving a clean gap between
    # the process arrows and the left spine of each histogram.
    for right in right_axes:
        position = right.get_position()
        shift = 0.018
        right.set_position(
            [position.x0 + shift, position.y0, position.width - shift, position.height]
        )

    left_position = left_axes[0].get_position()
    middle_left = min(axis.get_position().x0 for axis in middle_axes[0])
    middle_right = max(axis.get_position().x1 for axis in middle_axes[0])
    right_position = right_axes[0].get_position()
    left_center = (left_position.x0 + left_position.x1) / 2
    middle_center = (middle_left + middle_right) / 2
    right_center = (right_position.x0 + right_position.x1) / 2

    title_y = 0.965
    fig.text(
        left_center,
        title_y,
        "Compute spectral radius",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        middle_center,
        title_y,
        "Fit null model and simulate\nunder null",
        ha="center",
        va="center",
        fontsize=11.5,
        weight="bold",
        linespacing=1.05,
    )
    fig.text(
        right_center,
        title_y,
        "Compare spectral radius to null",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )

    column_gaps: list[float] = []
    for row in range(2):
        left_position = left_axes[row].get_position()
        middle_left = min(axis.get_position().x0 for axis in middle_axes[row])
        middle_right = max(axis.get_position().x1 for axis in middle_axes[row])
        right_position = right_axes[row].get_position()
        column_gaps.extend(
            [middle_left - left_position.x1, right_position.x0 - middle_right]
        )
    arrow_half_length = min(column_gaps) * 0.34

    for row in range(2):
        left_position = left_axes[row].get_position()
        middle_left = min(axis.get_position().x0 for axis in middle_axes[row])
        middle_right = max(axis.get_position().x1 for axis in middle_axes[row])
        right_position = right_axes[row].get_position()
        row_y = left_position.y0 + left_position.height / 2
        add_arrow(fig, left_position.x1, middle_left, row_y, arrow_half_length)
        add_arrow(fig, middle_right, right_position.x0, row_y, arrow_half_length)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiment_results/gnm_components/gnm_method_overview.png"))
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    result = build_figure(args.output, n_samples=args.samples, seed=args.seed)
    print(f"Wrote {result.resolve()}")


if __name__ == "__main__":
    main()
