"""
Chart: Private LB progression across all experiments (chronological).
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

methods = [
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
private_lb  = [0.8956, 0.8698, 0.8582, 0.8956, 0.8138, 0.7126, 0.7126,
               0.6095, 0.5854, 0.6410, 0.6430]
bar_colors  = [GREY, ORANGE, ORANGE, BLUE, BLUE, GREEN, GREEN,
               PURPLE, PURPLE, TEAL, TEAL]
x = np.arange(len(methods), dtype=float)

canvas = ChartCanvas(
    figsize=(16, 8),
    title="ShelfSense-M5: Kaggle Private LB Progression (Chronological)",
    ylabel="Private LB WRMSSE (lower is better)",
)
canvas.add_bars(x, private_lb, colors=bar_colors)
canvas.set_ylim(0.44, 1.15)
canvas.set_xticks(x, methods)

best_so_far = np.minimum.accumulate(private_lb)
canvas.add_step_line(
    np.append(x - 0.31, x[-1] + 0.31),
    np.append(best_so_far, best_so_far[-1]),
    color=RED, label="Best private LB to date",
)

for xc in [2.5, 6.5, 8.5]:
    canvas.add_phase_separator(xc)
canvas.add_phase_label(1.0, "Classical\n(1k sample)")
canvas.add_phase_label(4.5, "LightGBM\nglobal")
canvas.add_phase_label(7.5, "Multi-horizon")
canvas.add_phase_label(9.5, "Per-store")

# Three callouts — placed with "free" so they all land at the same y in the margin
# (bars at x=2,5,6 are well below cursor; no diagonal crossing)
canvas.add_callout(
    target_x=3, target_y=private_lb[3] + 0.015,
    text="Eval rows filled with SN28\nuntil recursive forecast added",
    placement="free", x_offset=2.0, color=BLUE,
)
canvas.add_callout(
    target_x=4, target_y=private_lb[4] + 0.015,
    text="Cross-series learning\n−35% vs SN28 baseline",
    placement="free", x_offset=5.5, color=BLUE,
)
canvas.add_callout(
    target_x=8, target_y=private_lb[8] + 0.015,
    text="Direct 28-step prediction\neliminates compounding  ★ best",
    placement="free", x_offset=8.0, color=PURPLE, fontweight="bold",
)

canvas.add_legend([
    mpatches.Patch(color=GREY,   label="Baseline"),
    mpatches.Patch(color=ORANGE, label="Classical (1k fill)"),
    mpatches.Patch(color=BLUE,   label="LightGBM global"),
    mpatches.Patch(color=GREEN,  label="Blend — per-category"),
    mpatches.Patch(color=PURPLE, label="Multi-horizon"),
    mpatches.Patch(color=TEAL,   label="Per-store"),
    plt.Line2D([0], [0], color=RED, lw=2, ls="--", label="Best private LB to date"),
], ncol=4)

canvas.save(os.path.join(CHARTS, "leaderboard_progression.png"))
