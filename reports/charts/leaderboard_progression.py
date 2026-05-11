"""
Chart: Private LB progression across all experiments (chronological).

Phase 1/2 history + Phase 3 Dagster-asset variants.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.shelfsense.visualization.charts import (
    ChartCanvas, BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, GREY, LGREY,
)

CHARTS = os.path.dirname(os.path.abspath(__file__))

# ── Phase 1/2 (left section) ──────────────────────────────────────────────────
GOLD = "#B8860B"

methods_12 = [
    "SN28\nbaseline",
    "ETS\n(1k fill)",
    "ARIMA\n(1k fill)",
    "LightGBM\n(SN28-filled\neval)",
    "LightGBM\nglobal\nrecursive",
    "Blend\n(per-category\n+ global)",
    "Blend\n(refined\nrecursive)",
    "Multi-horizon\nglobal",
    "Multi-horizon\nblend",
    "Per-store\n(standalone)",
    "Per-store\nblend",
]
private_12 = [0.8956, 0.8698, 0.8582, 0.8956, 0.8138, 0.7126, 0.7126,
              0.6095, 0.5854, 0.6410, 0.6430]
colors_12  = [GREY, ORANGE, ORANGE, BLUE, BLUE, GREEN, GREEN,
              PURPLE, PURPLE, TEAL, TEAL]

# ── Phase 3 Dagster model variants (right section) ───────────────────────────
methods_p3 = [
    "tvp=1.7 MH\n(model_tvp_17)",
    "per_dept\n(model_per_dept)",
    "RMSE-MH\n(model_rmse_mh)",
    "per_store\n(model_per_store)",
    "store×dept\n(model_store_dept)",
    "ylags\n(model_ylags)",
    "tvp=1.3 MH\n(model_tvp_13)",
]
private_p3 = [0.6623, 0.6137, 0.6205, 0.6410, 0.5882, 0.5749, 0.5693]
colors_p3  = [GOLD] * 7

methods = methods_12 + methods_p3
private_lb = private_12 + private_p3
bar_colors = colors_12 + colors_p3

x = np.arange(len(methods), dtype=float)

canvas = ChartCanvas(
    figsize=(22, 8),
    title="ShelfSense-M5: Kaggle Private LB Progression — Phase 1/2 → Phase 3 Dagster Variants",
    ylabel="Private LB WRMSSE (lower is better)",
)
canvas.add_bars(x, private_lb, colors=bar_colors)
canvas.set_ylim(0.44, 1.15)
canvas.set_xticks(x, methods, fontsize=8.5)

best_so_far = np.minimum.accumulate(private_lb)
canvas.add_step_line(
    np.append(x - 0.31, x[-1] + 0.31),
    np.append(best_so_far, best_so_far[-1]),
    color=RED, label="Best private LB to date",
)

# Phase separators
for xc in [2.5, 6.5, 8.5, 10.5]:
    canvas.add_phase_separator(xc)
canvas.add_phase_label(1.0, "Classical\n(1k sample)")
canvas.add_phase_label(4.5, "LightGBM\nglobal")
canvas.add_phase_label(7.5, "Multi-horizon")
canvas.add_phase_label(9.5, "Per-store")
canvas.add_phase_label(14.5, "Phase 3\nDagster assets")

canvas.add_callout(
    target_x=3, target_y=canvas.bar_top_for_arrow(3),
    text="Eval rows filled with SN28\nuntil recursive forecast added",
    placement="top", x_offset=-1.0, color=BLUE, same_row=True,
    connectionstyle="angle,angleA=0,angleB=-90",
)
canvas.add_callout(
    target_x=4, target_y=canvas.bar_top_for_arrow(4),
    text="Cross-series learning\n−35% vs SN28 baseline",
    placement="top", x_offset=+1.5, color=BLUE, same_row=True,
    connectionstyle="angle,angleA=180,angleB=-90",
)
canvas.add_callout(
    target_x=8, target_y=canvas.bar_top_for_arrow(8),
    text="Direct 28-step prediction\neliminates compounding  ★ Phase 2 best",
    placement="top", x_offset=0, color=PURPLE, fontweight="bold",
)
canvas.add_callout(
    target_x=len(methods) - 1, target_y=canvas.bar_top_for_arrow(len(methods) - 1),
    text="tvp=1.3 MH\n★ Final best (0.5693)\n36% reduction from SN28",
    placement="top", x_offset=0, color=GOLD, fontweight="bold",
)

canvas.add_legend([
    mpatches.Patch(color=GREY,   label="Baseline"),
    mpatches.Patch(color=ORANGE, label="Classical (1k fill)"),
    mpatches.Patch(color=BLUE,   label="LightGBM global"),
    mpatches.Patch(color=GREEN,  label="Blend — per-category"),
    mpatches.Patch(color=PURPLE, label="Multi-horizon"),
    mpatches.Patch(color=TEAL,   label="Per-store"),
    mpatches.Patch(color=GOLD,   label="Phase 3 Dagster variants"),
    plt.Line2D([0], [0], color=RED, lw=2, ls="--", label="Best private LB to date"),
], ncol=4)

canvas.save(os.path.join(CHARTS, "leaderboard_progression.png"))
