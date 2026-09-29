from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiment_results"
OUTPUT_PNG = RESULTS_DIR / "real_null_model_intersections_n1000.png"
OUTPUT_PDF = RESULTS_DIR / "real_null_model_intersections_n1000.pdf"

MODEL_COLUMNS = ["gnm", "configuration", "configuration_fast", "hidden"]
MODEL_LABELS = [
    r"$G(N,M)$",
    "Configuration-base",
    "Configuration-swap",
    "Chung--Lu",
]


def load_decisions() -> pd.DataFrame:
    paths = [
        RESULTS_DIR / "detail_grid_gnm.csv",
        RESULTS_DIR / "detail_grid_hidden.csv",
        RESULTS_DIR / "detail_grid.csv",
    ]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[(frame["dataset"] == "real") & (frame["n_samples"] == 1000)]
        frames.append(frame[["graph_type", "model", "reject_H0"]])

    decisions = pd.concat(frames, ignore_index=True).pivot(
        index="graph_type", columns="model", values="reject_H0"
    )
    decisions = decisions.reindex(columns=MODEL_COLUMNS)
    if decisions.isna().any().any():
        raise ValueError("Missing B=1000 decision for at least one real network/model pair")
    return decisions.astype(bool)


def main() -> None:
    decisions = load_decisions()
    intersections = decisions.value_counts().reset_index(name="count")
    intersections = intersections.sort_values(
        ["count", *MODEL_COLUMNS],
        ascending=[False, *([False] * len(MODEL_COLUMNS))],
        kind="stable",
    ).reset_index(drop=True)

    membership = intersections[MODEL_COLUMNS].to_numpy(dtype=bool).T
    counts = intersections["count"].to_numpy(dtype=int)
    x = np.arange(len(intersections))
    y = np.arange(len(MODEL_COLUMNS))

    blue = "#2F6DAE"
    pale_blue = "#D7E3F1"
    gray = "#D9D9D9"

    fig = plt.figure(figsize=(7.6, 4.7), dpi=220)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.35], hspace=0.06)
    ax_bar = fig.add_subplot(grid[0])
    ax_matrix = fig.add_subplot(grid[1], sharex=ax_bar)

    bars = ax_bar.bar(x, counts, width=0.64, color=blue, edgecolor="none")
    ax_bar.set_ylabel("Number of networks")
    ax_bar.set_ylim(0, max(counts) * 1.24)
    ax_bar.set_xticks([])
    ax_bar.spines[["top", "right", "bottom"]].set_visible(False)
    ax_bar.tick_params(axis="x", length=0)
    ax_bar.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax_bar.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            count + max(counts) * 0.025,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for column, xpos in enumerate(x):
        active_rows = np.flatnonzero(membership[:, column])
        if len(active_rows) > 1:
            ax_matrix.plot(
                [xpos, xpos],
                [active_rows.min(), active_rows.max()],
                color=blue,
                linewidth=1.8,
                zorder=1,
            )
        ax_matrix.scatter(
            np.full(len(y), xpos),
            y,
            s=43,
            color=np.where(membership[:, column], blue, gray),
            edgecolors="none",
            zorder=2,
        )

    for row in y:
        if row % 2 == 1:
            ax_matrix.axhspan(row - 0.5, row + 0.5, color=pale_blue, alpha=0.32, zorder=0)

    ax_matrix.set_yticks(y)
    ax_matrix.set_yticklabels(MODEL_LABELS, fontsize=9)
    ax_matrix.set_ylim(len(y) - 0.5, -0.5)
    ax_matrix.set_xlim(-0.6, len(intersections) - 0.4)
    ax_matrix.set_xticks(x)
    ax_matrix.set_xticklabels([])
    ax_matrix.set_xlabel("Nonempty intersections of rejection sets")
    ax_matrix.tick_params(axis="x", length=0)
    ax_matrix.tick_params(axis="y", length=0, pad=8)
    ax_matrix.spines[["top", "right", "bottom", "left"]].set_visible(False)

    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(intersections.to_string(index=False))


if __name__ == "__main__":
    main()
