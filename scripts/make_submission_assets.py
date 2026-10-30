#!/usr/bin/env python
"""Generate figures for the Week 1-5 submission report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "submission" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY = "#0F172A"
TEAL = "#0D9488"
ORANGE = "#D97706"
SLATE = "#475569"
MUTED = "#64748B"
LIGHT = "#F1F5F9"
BORDER = "#CBD5E1"
RED = "#B91C1C"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": SLATE,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "figure.dpi": 200,
    }
)


def _box(ax, x, y, w, h, text, face=LIGHT, edge=BORDER, color=NAVY, size=9.0, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
        linespacing=1.35,
    )
    return patch


def _arrow(ax, start, end, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.3,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    rows = [
        (44.0, "DATA", "SIFT .fvecs / .ivecs\n(official TEXMEX)", "Adapters + exact\nGT cache", "Benchmark harness\nparameter sweep"),
        (22.0, "INDEX CORE", "GraphStore\nvectors · IDs · adjacency · in-degrees", "HNSW\ninsert · search · prune · persist · repair", None),
        (3.0, "EVIDENCE", "Metrics\nRecall@K · p50/p95 · RSS · build", "result.json · summary.csv\nrecall/latency plots", None),
    ]

    for y, label, *boxes in rows:
        ax.add_patch(
            FancyBboxPatch(
                (6.0, y - 2.0),
                92.0,
                max(14.0, 7.0),
                boxstyle="round,pad=0.2,rounding_size=1.0",
                linewidth=0,
                facecolor="#F8FAFC",
                zorder=0,
            )
        )
        ax.text(2.8, y + 3.0, label, rotation=90, ha="center", va="center", fontsize=8, color=MUTED, fontweight="bold")

    # Data row
    _box(ax, 9.0, 45.0, 25.0, 9.0, rows[0][2], size=8.6)
    _box(ax, 39.5, 45.0, 25.0, 9.0, rows[0][3], size=8.6)
    _box(ax, 70.0, 45.0, 24.0, 9.0, rows[0][4], size=8.6)
    _arrow(ax, (34.6, 49.5), (39.0, 49.5))
    _arrow(ax, (65.1, 49.5), (69.5, 49.5))

    # Core row
    core = FancyBboxPatch(
        (9.0, 24.0),
        85.0,
        10.0,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        linewidth=1.4,
        edgecolor=TEAL,
        facecolor="#ECFDF5",
    )
    ax.add_patch(core)
    ax.text(51.5, 31.2, "from-scratch Python / NumPy core", ha="center", va="center", fontsize=7.6, color=TEAL, fontweight="bold")
    _box(ax, 11.5, 24.6, 36.0, 5.6, "GraphStore — vectors · IDs · adjacency · in-degrees", face="white", edge=TEAL, size=8.4)
    _box(ax, 55.5, 24.6, 36.0, 5.6, "HNSW — insert · search · prune · persist · repair", face="white", edge=TEAL, size=8.4)
    ax.annotate("", xy=(54.8, 27.4), xytext=(47.9, 27.4), arrowprops=dict(arrowstyle="<|-|>", color=TEAL, lw=1.4, mutation_scale=10))

    # Evidence row
    _box(ax, 9.0, 4.0, 35.0, 8.2, rows[2][2], size=8.6)
    _box(ax, 54.0, 4.0, 40.0, 8.2, rows[2][3], size=8.6)
    _arrow(ax, (44.4, 8.1), (53.5, 8.1))

    # Vertical flow
    _arrow(ax, (51.5, 44.5), (51.5, 34.6), color=TEAL)
    _arrow(ax, (51.5, 23.6), (51.5, 12.6), color=TEAL)
    ax.text(53.0, 41.7, "build / search", fontsize=7.4, color=MUTED, style="italic")
    ax.text(53.0, 20.7, "measure", fontsize=7.4, color=MUTED, style="italic")

    fig.tight_layout(pad=0.4)
    fig.savefig(ASSETS / "architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flow() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch((1.5, 1.0), 46.0, 47.0, boxstyle="round,pad=0.3,rounding_size=1.2", linewidth=1.0, edgecolor=BORDER, facecolor="#F8FAFC")
    )
    ax.add_patch(
        FancyBboxPatch((52.5, 1.0), 46.0, 47.0, boxstyle="round,pad=0.3,rounding_size=1.2", linewidth=1.0, edgecolor=BORDER, facecolor="#F8FAFC")
    )
    ax.text(24.5, 45.2, "BUILD — per inserted vector", ha="center", fontsize=10.5, color=NAVY, fontweight="bold")
    ax.text(75.5, 45.2, "SEARCH — per query", ha="center", fontsize=10.5, color=NAVY, fontweight="bold")

    build_steps = [
        "1. Sample level  ~ floor(-ln U / ln M)",
        "2. Greedy descent from entry point",
        "3. ef-bounded search on each layer",
        "4. Select diverse neighbours",
        "5. Link both ways; prune if full",
    ]
    y = 36.8
    for index, text in enumerate(build_steps):
        _box(ax, 4.5, y, 40.0, 5.6, text, size=8.2, color=NAVY)
        if index < len(build_steps) - 1:
            _arrow(ax, (24.5, y - 0.1), (24.5, y - 2.0), color=TEAL)
        y -= 7.6
    ax.text(24.5, 3.5, "repeat for every vector", ha="center", fontsize=7.6, color=MUTED, style="italic")

    search_steps = [
        "1. Start at the global entry point",
        "2. Greedy descent through upper layers",
        "3. ef-bounded best-first search on layer 0",
        "4. Return the k closest candidates",
    ]
    y = 34.0
    for index, text in enumerate(search_steps):
        _box(ax, 55.5, y, 40.0, 5.6, text, size=8.2, color=NAVY)
        if index < len(search_steps) - 1:
            _arrow(ax, (75.5, y - 0.1), (75.5, y - 2.0), color=TEAL)
        y -= 7.6
    ax.text(75.5, 6.5, "layer 0: 2M links · upper layers: M", ha="center", fontsize=7.6, color=MUTED, style="italic")

    fig.tight_layout(pad=0.4)
    fig.savefig(ASSETS / "hnsw_flow.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def results() -> None:
    result = json.loads((ROOT / "reports" / "phase1" / "e1_sift100k" / "result.json").read_text())
    custom = {row["ef"]: row for row in result["churnvec"]}
    lib = {row["ef"]: row for row in result["hnswlib"]}
    efs = sorted(custom)
    recall_c = [custom[ef]["recall"] for ef in efs]
    recall_l = [lib[ef]["recall"] for ef in efs]
    p50_c = [custom[ef]["p50_ms"] for ef in efs]
    p50_l = [lib[ef]["p50_ms"] for ef in efs]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    ax = axes[0]
    ax.plot(efs, recall_c, marker="o", color=TEAL, linewidth=2.2, markersize=5, label="ChurnVec")
    ax.plot(efs, recall_l, marker="s", color=ORANGE, linewidth=1.8, linestyle="--", markersize=4.5, label="hnswlib 0.8.0")
    ax.axhline(0.90, color=RED, linewidth=1.0, linestyle=":", label="gate target 0.90")
    ax.set_xscale("log", base=2)
    ax.set_xticks(efs)
    ax.set_xticklabels([str(ef) for ef in efs])
    ax.set_xlabel("efSearch")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.84, 1.005)
    ax.set_title("Recall vs efSearch (SIFT100K)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.annotate(
        "0.9846 vs 0.9851\nat ef=64",
        xy=(64, custom[64]["recall"]),
        xytext=(90, 0.915),
        fontsize=7.8,
        color=NAVY,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9),
    )

    ax = axes[1]
    ax.plot(efs, p50_c, marker="o", color=TEAL, linewidth=2.2, markersize=5, label="ChurnVec")
    ax.plot(efs, p50_l, marker="s", color=ORANGE, linewidth=1.8, linestyle="--", markersize=4.5, label="hnswlib 0.8.0")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(efs)
    ax.set_xticklabels([str(ef) for ef in efs])
    ax.set_xlabel("efSearch")
    ax.set_ylabel("p50 latency (ms, log scale)")
    ax.set_title("Query latency vs efSearch")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.annotate(
        "0.978 vs 0.155 ms",
        xy=(64, custom[64]["p50_ms"]),
        xytext=(90, 0.36),
        fontsize=7.8,
        color=NAVY,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9),
    )

    fig.text(
        0.5,
        -0.01,
        "SIFT100K · 10,000 queries · exact recomputed ground truth · M=16, efConstruction=100 · single thread",
        ha="center",
        fontsize=7.6,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ASSETS / "results_e1.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pruning_experiment() -> None:
    labels = ["keep_pruned=True", "keep_pruned=False"]
    build = [94.2, 46.0]
    colors = [RED, TEAL]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axes[0]
    bars = ax.barh(positions, build, color=colors, height=0.5)
    for bar, value in zip(bars, build):
        ax.text(value + 2, bar.get_y() + bar.get_height() / 2, f"{value:.1f} s", va="center", fontsize=9, color=NAVY)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Build time (s)")
    ax.set_title("20K SIFT build time")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax = axes[1]
    unreachable = [10813, 0]
    bars = ax.barh(positions, unreachable, color=colors, height=0.5)
    ax.text(250, positions[0], "10,813 / 20,000 unreachable", va="center", fontsize=8.5, color=WHITE, fontweight="bold")
    ax.text(250, positions[1], "0 unreachable", va="center", fontsize=9, color=TEAL, fontweight="bold")
    ax.set_xlim(0, 13000)
    ax.set_xlabel("Unreachable nodes after build")
    ax.set_title("Connectivity at 20K")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_yticks(positions)
    ax.set_yticklabels([])
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(ASSETS / "pruning_experiment.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    flow()
    results()
    pruning_experiment()
    for path in sorted(ASSETS.glob("*.png")):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")
