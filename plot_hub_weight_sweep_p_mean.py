from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiment_results"


def plot_one(directory: Path) -> None:
    summary = pd.read_csv(directory / "hub_weight_sweep_summary.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=220)

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
    ax.set_xlabel("number of hub nodes", fontsize=15)
    ax.set_ylabel("mean p-value", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(
        fontsize=12,
        markerscale=1.0,
        ncol=2,
        loc="upper right",
        borderaxespad=0.5,
        frameon=True,
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(directory / "hub_weight_sweep_p_mean.png", bbox_inches="tight")
    plt.close(fig)


def plot_fixed_hub_count_comparison() -> None:
    settings = [
        (
            RESULTS / "hub_weight_sweep_N50_M100_gnm_50seed",
            (5, 10, 15),
            r"$(N,M)=(50,100)$",
        ),
        (
            RESULTS / "hub_weight_sweep_N100_M200_gnm_50seed",
            (10, 20, 30),
            r"$(N,M)=(100,200)$",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=220, sharey=True)
    for ax, (directory, hub_counts, panel_title) in zip(axes, settings):
        summary = pd.read_csv(directory / "hub_weight_sweep_summary.csv")
        for n_hubs in hub_counts:
            group = summary[summary["n_hubs"] == n_hubs].sort_values("hub_weight")
            ax.plot(
                group["hub_weight"],
                group["p_mean"],
                marker="o",
                markersize=6,
                linewidth=2.2,
                label=rf"$k={n_hubs}$",
            )

        ax.axhline(0.05, color="#555555", linestyle="--", linewidth=1.4)
        ax.set_xlabel(r"central-node weight $w$", fontsize=14)
        ax.set_xticks(sorted(summary["hub_weight"].unique()))
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=11, loc="upper right", frameon=True)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("mean p-value", fontsize=14)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(RESULTS / "hub_weight_sweep_fixed_k_p_mean.png", bbox_inches="tight")
    fig.savefig(RESULTS / "hub_weight_sweep_fixed_k_p_mean.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for name in (
        "hub_weight_sweep_N50_M100_gnm_50seed",
        "hub_weight_sweep_N100_M200_gnm_50seed",
    ):
        plot_one(RESULTS / name)
    plot_fixed_hub_count_comparison()


if __name__ == "__main__":
    main()
