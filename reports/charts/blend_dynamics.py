"""
Chart: Ensemble diversity — when blending helped (per-category) vs hurt (per-store).

Two-panel figure using ChartCanvas.from_axes().
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.shelfsense.visualization.charts import (
    ChartCanvas, BLUE, GREEN, RED, TEAL, TEAL_LIGHT, LGREY,
)

CHARTS = os.path.dirname(os.path.abspath(__file__))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))
fig.patch.set_facecolor("white")
fig.suptitle("Ensemble Diversity: When Blending Helps vs Hurts",
             fontsize=14, fontweight="bold", y=1.01)

# ── Panel A: per-category + global — blend HELPED ────────────────────────────
canvas_a = ChartCanvas.from_axes(
    ax1, fig,
    title="When blending helped\n(per-category + global)",
    ylabel="Private LB WRMSSE (lower is better)",
    title_color=GREEN,
)

labels_a   = ["Global recursive\nval WRMSSE: 0.5422",
               "Blend\n(0.6×per-cat + 0.4×global)\nval WRMSSE: 0.5545"]
private_a  = np.array([0.8138, 0.7126])
canvas_a.add_bars(np.arange(2, dtype=float), private_a,
                  colors=[BLUE, GREEN], width=0.45, value_size=11, value_pad=0.015)
canvas_a.set_ylim(0.60, 0.97)
canvas_a.set_xticks(np.arange(2), labels_a, fontsize=11)

canvas_a.add_callout(
    target_x=1, target_y=private_a[1] + 0.015,
    text="Blend −0.101\non private LB",
    # "free" positions text at y=0.930, clear of the panel title
    placement="free", x_offset=1.0, y_offset=-0.015,
    color=GREEN, fontweight="bold", fontsize=10,
)
canvas_a.ax.legend(handles=[
    mpatches.Patch(color=BLUE,  label="Global recursive"),
    mpatches.Patch(color=GREEN, label="Blend"),
], fontsize=9, loc="lower right")

# ── Panel B: per-store + global — blend HURT ─────────────────────────────────
canvas_b = ChartCanvas.from_axes(
    ax2, fig,
    title="When blending hurt\n(per-store + global recursive)",
    ylabel="Private LB WRMSSE (lower is better)",
    title_color=RED,
)

labels_b  = ["Per-store alone\nval WRMSSE: 0.6140",
              "Blend\n(0.6×per-store + 0.4×global)\nval WRMSSE: 0.5737"]
private_b = np.array([0.6410, 0.6430])
canvas_b.add_bars(np.arange(2, dtype=float), private_b,
                  colors=[TEAL, TEAL_LIGHT], width=0.45, value_size=11, value_pad=0.003)
canvas_b.set_ylim(0.60, 0.97)
canvas_b.set_xticks(np.arange(2), labels_b, fontsize=11)

canvas_b.add_callout(
    target_x=1, target_y=private_b[1] + 0.003,
    text="Blend +0.002\n(global recursive\nadds noise, not signal)",
    # "free" positions text at y=0.860, well below the panel title
    placement="free", x_offset=1.0, y_offset=-0.085,
    color=RED, fontweight="bold", fontsize=10,
)
canvas_b.ax.legend(handles=[
    mpatches.Patch(color=TEAL,       label="Per-store alone"),
    mpatches.Patch(color=TEAL_LIGHT, label="Blend"),
], fontsize=9, loc="lower right")

# Validate both panels before saving
for label, cv in [("Panel A", canvas_a), ("Panel B", canvas_b)]:
    violations = cv.validate()
    if violations:
        raise ValueError(f"{label} violations:\n" + "\n".join(f"  {v}" for v in violations))

fig.text(
    0.5, -0.06,
    "First case: global and per-category were comparably imperfect → diversity gain on private LB\n"
    "Second case: per-store already outperformed global recursive → blending in the weaker component added noise",
    ha="center", fontsize=9.5, color="#444444", fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8", edgecolor=LGREY, alpha=0.9),
)

ChartCanvas.save_fig(fig, os.path.join(CHARTS, "blend_dynamics.png"), bottom_adjust=0.18)
