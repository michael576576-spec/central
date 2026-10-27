from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiment_results"
LABEL_FONTSIZE = 17
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 14


def draw_hub_count_sweep(ax, summary: pd.DataFrame, *, show_xlabel: bool = True) -> None:
    for weight, group in summary.groupby("hub_weight", sort=True):
        group = group.sort_values("n_hubs")
        ax.plot(
            group["n_hubs"],
            group["p_mean"],
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            label=f"w={weight:g}",
        )

    ax.axhline(0.05, color="#555555", linestyle="--", linewidth=1.4)
    ax.set_ylim(top=0.55)
    if show_xlabel:
        ax.set_xlabel("number of hub nodes", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("mean p-value", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        markerscale=1.0,
        ncol=2,
        loc="upper right",
        borderaxespad=0.5,
        frameon=True,
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_one(directory: Path) -> None:
    summary = pd.read_csv(directory / "hub_weight_sweep_summary.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=220)
    draw_hub_count_sweep(ax, summary)
    fig.tight_layout()
    fig.savefig(directory / "hub_weight_sweep_p_mean.png", bbox_inches="tight")
    plt.close(fig)


def plot_stacked_hub_count_comparison() -> None:
    directories = (
        RESULTS / "hub_weight_sweep_N50_M100_gnm_50seed",
        RESULTS / "hub_weight_sweep_N100_M200_gnm_50seed",
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.6), dpi=220)
    for index, (ax, directory) in enumerate(zip(axes, directories)):
        summary = pd.read_csv(directory / "hub_weight_sweep_summary.csv")
        draw_hub_count_sweep(ax, summary, show_xlabel=index == 1)
    fig.tight_layout()
    output = RESULTS / "figure_relayouts" / "figure3_stacked.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def draw_fixed_hub_count(ax, summary: pd.DataFrame, hub_counts: tuple[int, ...], *, show_xlabel: bool = True, show_ylabel: bool = True) -> None:
    for n_hubs in hub_counts:
        group = summary[summary["n_hubs"] == n_hubs].sort_values("hub_weight")
        ax.plot(
            group["hub_weight"],
            group["p_mean"],
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            label=rf"$k={n_hubs}$",
        )

    ax.axhline(0.05, color="#555555", linestyle="--", linewidth=1.4)
    if show_xlabel:
        ax.set_xlabel(r"central-node weight $w$", fontsize=LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("mean p-value", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(sorted(summary["hub_weight"].unique()))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=True)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_fixed_hub_count_comparison() -> None:
    settings = (
        (
            RESULTS / "hub_weight_sweep_N50_M100_gnm_50seed",
            (5, 10, 15),
        ),
        (
            RESULTS / "hub_weight_sweep_N100_M200_gnm_50seed",
            (10, 20, 30),
        ),
    )

    for directory, hub_counts in settings:
        summary = pd.read_csv(directory / "hub_weight_sweep_summary.csv")
        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=220)
        draw_fixed_hub_count(ax, summary, hub_counts)

        fig.tight_layout()
        setting_name = directory.name.removeprefix("hub_weight_sweep_").removesuffix("_gnm_50seed")
        output_name = f"hub_weight_sweep_fixed_k_{setting_name}_p_mean.pdf"
        fig.savefig(RESULTS / output_name, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.6), dpi=220)
    for index, (ax, (directory, hub_counts)) in enumerate(zip(axes, settings)):
        summary = pd.read_csv(directory / "hub_weight_sweep_summary.csv")
        draw_fixed_hub_count(ax, summary, hub_counts, show_xlabel=index == 1)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_relayouts" / "figure4_stacked.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=220, sharey=True)
    for index, (ax, (directory, hub_counts)) in enumerate(zip(axes, settings)):
        summary = pd.read_csv(directory / "hub_weight_sweep_summary.csv")
        draw_fixed_hub_count(ax, summary, hub_counts, show_ylabel=index == 0)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(RESULTS / "hub_weight_sweep_fixed_k_p_mean.png", bbox_inches="tight")
    fig.savefig(RESULTS / "hub_weight_sweep_fixed_k_p_mean.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for name in (
        "hub_weight_sweep_N50_M100_gnm_50seed",
        "hub_weight_sweep_N100_M200_gnm_50seed",
        "hub_weight_sweep_N50_M100_K1_50_gnm_50seed",
    ):
        plot_one(RESULTS / name)
    plot_stacked_hub_count_comparison()
    plot_fixed_hub_count_comparison()


if __name__ == "__main__":
    main()
