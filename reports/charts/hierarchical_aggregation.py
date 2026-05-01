"""
Chart: Top-down vs bottom-up Prophet on 1k sample.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.patches as mpatches

from src.shelfsense.visualization.charts import (
    ChartCanvas, BLUE, ORANGE, GREEN, GREY,
)

CHARTS = os.path.dirname(os.path.abspath(__file__))

labels = ["Bottom-Up\nProphet", "Top-Down\nNational", "Top-Down\nState",
          "Top-Down\nDept", "Top-Down\nCategory"]
values = np.array([0.6638, 0.5580, 0.5740, 0.5565, 0.5555])
colors = [ORANGE, BLUE, BLUE, BLUE, GREEN]
x5     = np.arange(len(labels), dtype=float)

canvas = ChartCanvas(figsize=(9, 6),
                     title="Top-Down vs Bottom-Up: Prophet on 1k Sample",
                     ylabel="WRMSSE (lower is better)")

canvas.add_bars(x5, values, colors=colors, width=0.55, value_size=11, value_pad=0.007)
canvas.set_ylim(0.47, 0.78)
canvas.set_xticks(x5, labels, fontsize=11)

canvas.add_hline(0.6778, color=GREY, label="SN28 ref (0.6778)")

# Callout — arc arrow from upper-center to Top-Down Category bar top
# cursor ≈ 0.78 − 0.31×0.18×0.38 ≈ 0.7588; original text y was 0.735
canvas.add_callout(
    target_x=4.0, target_y=canvas.bar_top_for_arrow(4.0),
    text="−0.108 vs Bottom-Up\n(category aggregation\ncancels sparse-series noise)",
    placement="free", x_offset=1.5, y_offset=-0.024,
    color=GREEN, fontweight="bold", fontsize=9.5,
    connectionstyle="arc3,rad=-0.15",
)

canvas.ax.legend(handles=[
    mpatches.Patch(color=ORANGE, label="Bottom-Up"),
    mpatches.Patch(color=BLUE,   label="Top-Down (other levels)"),
    mpatches.Patch(color=GREEN,  label="Top-Down Category (best)"),
], fontsize=10, loc="upper right")

canvas.save(os.path.join(CHARTS, "hierarchical_aggregation.png"), bottom_adjust=0.10)
