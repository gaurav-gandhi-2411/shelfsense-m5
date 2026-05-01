"""
Chart: Per-category WRMSSE — Baseline → Classical → LightGBM.

Three grouped bars per category. ETS HOBBIES bar is clipped at 2.80
with an overflow label showing the true value (3.27).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.patches as mpatches

from src.shelfsense.visualization.charts import (
    ChartCanvas, BLUE, ORANGE, GREEN, RED, GREY,
)

CHARTS = os.path.dirname(os.path.abspath(__file__))

categories  = ["FOODS", "HOUSEHOLD", "HOBBIES"]
sn28_scores = np.array([0.6400, 1.1580, 1.5949])
ets_scores  = np.array([0.5616, 1.7023, 3.2663])
lgbm_scores = np.array([0.5204, 0.5905, 0.6112])

CLIP = 2.80
x    = np.arange(len(categories), dtype=float)
w    = 0.24

canvas = ChartCanvas(figsize=(11, 7),
                     title="Per-Category WRMSSE: Baseline → Classical → LightGBM",
                     ylabel="WRMSSE (lower is better)")

canvas.add_bars(x - w, sn28_scores, colors=[GREY]   * 3, width=w, value_size=9.5)
canvas.add_bars(x,     np.minimum(ets_scores, CLIP),
                colors=[ORANGE] * 3, width=w, value_size=9.5, value_labels=False)
canvas.add_bars(x + w, lgbm_scores, colors=[BLUE]   * 3, width=w, value_size=9.5)

# Manual ETS value labels (clipped bar needs special handling)
for i, v in enumerate(ets_scores):
    if v > CLIP:
        canvas.add_bar_label(float(x[i]), CLIP, "3.27 ↑", color=ORANGE, pad=0.06)
    else:
        canvas.add_bar_label(float(x[i]), v, f"{v:.4f}", color=ORANGE, pad=0.012)

canvas.set_ylim(0, 3.35)
canvas.set_xticks(x, categories, fontsize=13)

# Clip reference line
canvas.add_hline(CLIP, color=ORANGE, lw=1.0, ls=":", alpha=0.5)
canvas.ax.text(0.02, CLIP + 0.04, f"chart clipped at {CLIP}", color=ORANGE,
               fontsize=8, fontstyle="italic",
               transform=canvas.ax.get_yaxis_transform())

# Callouts — x_offset spreads the text boxes so they don't overlap;
# both targets use bar_top_for_arrow() so arrow tips clear the value-label zone.
# ETS callout arrow: (1.5, cursor) → (2.0, safe_y) — short diagonal above all bars.
# LightGBM callout arrow: (2.94, cursor) → (2.24, safe_y) — short diagonal above all bars.
canvas.add_callout(
    target_x=float(x[2]),           # ETS HOBBIES bar center (clipped at CLIP)
    target_y=canvas.bar_top_for_arrow(float(x[2])),
    text="Zero-forecast fallback\n(390/1k sparse series)",
    placement="top", x_offset=-0.5,
    color=ORANGE, fontsize=9.5,
)
canvas.add_callout(
    target_x=float(x[2]) + w,       # LightGBM HOBBIES bar center
    target_y=canvas.bar_top_for_arrow(float(x[2]) + w),
    text="5× improvement\n3.27 → 0.61\n(cross-series signal)",
    placement="top", x_offset=0.7,
    color=RED, fontsize=9.5, fontweight="bold",
)

canvas.ax.legend(handles=[
    mpatches.Patch(color=GREY,   label="SN28 baseline"),
    mpatches.Patch(color=ORANGE, label="ETS (1k sample)"),
    mpatches.Patch(color=BLUE,   label="LightGBM Tweedie (full)"),
], fontsize=10, loc="upper left")

canvas.save(os.path.join(CHARTS, "per_category_journey.png"), bottom_adjust=0.08)
