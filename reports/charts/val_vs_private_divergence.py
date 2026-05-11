"""
Chart: Val WRMSSE delta vs private LB delta — Phase 3 model variants.

Shows 4 of 5 non-baseline variants improved validation WRMSSE but worsened
private LB (inversions). Standalone PNG version of notebook/02_failure_analysis
cell 4, rendered with matplotlib Agg for CI compatibility.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CHARTS = os.path.dirname(os.path.abspath(__file__))

# ── Data (from reports/leaderboard.md) ───────────────────────────────────────
# Baseline: tvp_13 val=0.6860, private=0.5693
BASELINE_VAL  = 0.6860
BASELINE_PRIV = 0.5693

variants = [
    ("tvp_13",     0.6860, 0.5693, "black",   "D", 140),   # baseline
    ("ylags",      0.6830, 0.5749, "crimson",  "o", 100),   # inversion
    ("store_dept", 0.6294, 0.5882, "crimson",  "o", 100),   # inversion
    ("per_dept",   0.7333, 0.6137, "#E07B39",  "s", 100),   # both worse
    ("rmse_mh",    0.6699, 0.6205, "crimson",  "o", 100),   # inversion
    ("per_store",  0.6140, 0.6410, "crimson",  "o", 100),   # inversion
]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")

# Quadrant shading
ax.fill_between([-0.12, 0], [0.1, 0.1], 0,
                alpha=0.07, color="red")        # Q2: inversion zone
ax.fill_between([0, 0.07], [0.1, 0.1], 0,
                alpha=0.05, color="orange")     # Q3: both worse
ax.fill_between([-0.12, 0], [0, 0], [-0.01, -0.01],
                alpha=0.06, color="green")      # Q4: ideal

ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

ax.text(-0.06, 0.091, "Val better,\nprivate worse\n(INVERSION)",
        ha="center", va="top", fontsize=9, color="#990000", style="italic")
ax.text(0.038, 0.091, "Both worse",
        ha="center", va="top", fontsize=9, color="darkorange", style="italic")
ax.text(-0.06, -0.003, "Both better (ideal)",
        ha="center", va="bottom", fontsize=9, color="green", style="italic")

# Label offsets to avoid overlap
offsets = {
    "tvp_13":     (6, -14),
    "ylags":      (6,   5),
    "store_dept": (-82,  5),
    "per_dept":   (6,   5),
    "rmse_mh":    (6,   5),
    "per_store":  (6, -14),
}

for name, val, priv, color, marker, size in variants:
    vd = val  - BASELINE_VAL
    pd = priv - BASELINE_PRIV
    ax.scatter(vd, pd, color=color, marker=marker, s=size, zorder=5,
               edgecolors="white" if name != "tvp_13" else "black", linewidths=0.8)
    dx, dy = offsets.get(name, (6, 5))
    ax.annotate(
        name, (vd, pd), xytext=(dx, dy), textcoords="offset points",
        fontsize=10,
        fontweight="bold" if name == "tvp_13" else "normal",
        color=color,
    )

ax.set_xlabel(
    "Val WRMSSE delta vs tvp_13  (negative = improved validation)", fontsize=11)
ax.set_ylabel(
    "Private LB delta vs tvp_13  (positive = worsened private LB)", fontsize=11)
ax.set_title(
    "Single-holdout validation overfits to one 28-day window\n"
    "4 of 5 non-baseline variants improved val but worsened private LB",
    fontsize=12, fontweight="bold",
)

legend_handles = [
    mpatches.Patch(color="crimson",    alpha=0.8,
                   label="Inversion (val better, private worse)"),
    mpatches.Patch(color="#E07B39",    alpha=0.8,
                   label="Consistent degradation (both worse)"),
    mpatches.Patch(color="black",      alpha=0.8,
                   label="Baseline (tvp_13)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
ax.set_xlim(-0.12, 0.07)
ax.set_ylim(-0.01, 0.10)

plt.tight_layout()
out = os.path.join(CHARTS, "val_vs_private_divergence.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
