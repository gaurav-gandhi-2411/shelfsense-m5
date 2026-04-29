"""
Generate portfolio charts for README.

Charts produced:
  reports/charts/leaderboard_progression.png  — private LB over time with inflection annotations
  reports/charts/per_category_journey.png     — FOODS/HOUSEHOLD/HOBBIES across key methods
  reports/charts/blend_dynamics.png           — when blending helped (Day 7) vs hurt (Day 10)
  reports/charts/hierarchical_aggregation.png — top-down vs bottom-up Prophet (unchanged)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS = os.path.join(PROJ_ROOT, "reports", "charts")
os.makedirs(CHARTS, exist_ok=True)

BLUE   = "#2C6FAC"
ORANGE = "#E07B39"
GREEN  = "#3A8C5C"
RED    = "#C0392B"
PURPLE = "#7B3FA0"
GREY   = "#888888"
TEAL   = "#1A7F7A"


# ── Chart 1: Private LB Progression (chronological) ──────────────────────────

fig, ax = plt.subplots(figsize=(13, 5.5))

methods = [
    "SN28\nbaseline",
    "ETS\n(1k fill)",
    "ARIMA\n(1k fill)",
    "LightGBM\nDay 6\n(SN28 eval)",
    "LightGBM\nglobal\nrecursive",
    "Blend\nDay 7\n(per-cat)",
    "Blend\nDay 8\n(v2 audit)",
    "MH global\nDay 9",
    "MH blend\nDay 9",
    "Per-store\nDay 10",
    "Per-store\nblend Day 10",
]
private_lb = [0.8956, 0.8698, 0.8582, 0.8956, 0.8138, 0.7126, 0.7126, 0.6095, 0.5854, 0.6410, 0.6430]
x = np.arange(len(methods))

# Color by type
colors = [
    GREY,          # SN28
    ORANGE,        # ETS
    ORANGE,        # ARIMA
    BLUE,          # LightGBM Day 6
    BLUE,          # global recursive
    GREEN,         # blend Day 7
    GREEN,         # blend Day 8
    PURPLE,        # MH global Day 9
    PURPLE,        # MH blend Day 9
    TEAL,          # per-store Day 10
    TEAL,          # per-store blend Day 10
]

bars = ax.bar(x, private_lb, color=colors, width=0.65, zorder=3, edgecolor='white', linewidth=0.5)
for rect, v in zip(bars, private_lb):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.006,
            f"{v:.4f}", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color='#333333')

# Best-to-date line
best_so_far = []
cur_best = 999
for v in private_lb:
    cur_best = min(cur_best, v)
    best_so_far.append(cur_best)
ax.step(x - 0.33, best_so_far, where='post', color=RED, lw=1.8, ls='--', zorder=6, label="Best private LB")
ax.plot(x[-1] + 0.33, best_so_far[-1], 'D', color=RED, ms=6, zorder=7)

# Inflection annotations
ax.annotate("LightGBM\ncross-series\n−35% from baseline",
            xy=(4, 0.8138), xytext=(3.5, 0.72),
            fontsize=8, color=BLUE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.3))

ax.annotate("Recursive bug fix\n(SN28 eval removed)",
            xy=(4, 0.8138), xytext=(4.3, 0.87),
            fontsize=7.5, color=BLUE, style='italic',
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))

ax.annotate("Multi-horizon\neliminates compounding\n−0.127 vs Day 8",
            xy=(8, 0.5854), xytext=(7.0, 0.50),
            fontsize=8, color=PURPLE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.3))

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8.2)
ax.set_ylabel("Private LB WRMSSE (lower is better)", fontsize=11)
ax.set_title("ShelfSense-M5: Private LB Progression (Chronological)", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0.45, 0.97)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3, zorder=0)

handles = [
    mpatches.Patch(color=GREY,   label="Baseline"),
    mpatches.Patch(color=ORANGE, label="Classical (1k-fill)"),
    mpatches.Patch(color=BLUE,   label="LightGBM global"),
    mpatches.Patch(color=GREEN,  label="LightGBM blend (per-cat)"),
    mpatches.Patch(color=PURPLE, label="Multi-horizon (Day 9)"),
    mpatches.Patch(color=TEAL,   label="Per-store (Day 10)"),
]
ax.legend(handles=handles, fontsize=8, loc='upper right', ncol=2)

plt.tight_layout()
path = os.path.join(CHARTS, "leaderboard_progression.png")
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Chart 2: Per-category journey ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

categories   = ["FOODS", "HOUSEHOLD", "HOBBIES"]
sn28_scores  = [0.6400,  1.1580,      1.5949]
ets_scores   = [0.5616,  1.7023,      3.2663]
lgbm_scores  = [0.5204,  0.5905,      0.6112]

x     = np.arange(len(categories))
width = 0.22

b1 = ax.bar(x - width,     sn28_scores, width, color=GREY,   label="SN28 baseline",         zorder=3)
b2 = ax.bar(x,             ets_scores,  width, color=ORANGE, label="ETS (1k sample)",        zorder=3, alpha=0.85, hatch='////')
b3 = ax.bar(x + width,     lgbm_scores, width, color=BLUE,   label="LightGBM Tweedie (full)",zorder=3)

for bars_group, scores, color in [(b1, sn28_scores, GREY), (b2, ets_scores, ORANGE), (b3, lgbm_scores, BLUE)]:
    for rect, v in zip(bars_group, scores):
        ypos = min(v, 2.58)
        label = f"{v:.2f}" if v > 2.5 else f"{v:.4f}"
        if v > 2.5:
            label += "\n(clipped)"
        ax.text(rect.get_x() + rect.get_width()/2, ypos + 0.04,
                label, ha='center', va='bottom', fontsize=7.8, color=color, fontweight='bold')

ax.annotate("5× improvement\n3.27 → 0.61\n(cross-series signal)",
            xy=(x[2] + width, lgbm_scores[2]),
            xytext=(x[2] + width + 0.35, 1.3),
            fontsize=9, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

ax.annotate("Zero-fallback\ncollapse",
            xy=(x[2], min(ets_scores[2], 2.58)),
            xytext=(x[2] - 0.35, 2.2),
            fontsize=8.5, color=ORANGE, style='italic',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylabel("WRMSSE (lower is better)", fontsize=11)
ax.set_title("Per-Category WRMSSE: Baseline → Classical → LightGBM", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, 2.8)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
path = os.path.join(CHARTS, "per_category_journey.png")
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Chart 3: Blend dynamics — helped vs hurt ──────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Ensemble Diversity: When Blending Helps vs Hurts", fontsize=13, fontweight='bold', y=1.01)

# ---- Panel A: Day 7 — blend HELPED ----
ax = axes[0]
ax.set_title("Day 7: Blending Helped\n(per-category + global)", fontsize=11, fontweight='bold', color=GREEN)

labels_7   = ["Global\nalone", "Per-cat\nalone", "Blend\n(0.6×pc+0.4×g)"]
val_7      = [0.5422,  0.5726,  0.5545]
private_7  = [0.8138,  None,    0.7126]

x7 = np.arange(3)
bars_val = ax.bar(x7 - 0.18, val_7, 0.33, color=[BLUE, GREEN, GREEN], alpha=0.7,
                  label="Val WRMSSE", zorder=3, edgecolor='white')
priv_vals = [0.8138, None, 0.7126]
priv_colors = [BLUE, None, GREEN]
for i, (pv, pc) in enumerate(zip(priv_vals, priv_colors)):
    if pv is not None:
        ax.bar(x7[i] + 0.18, pv, 0.33, color=pc, alpha=1.0, label="Private LB" if i == 0 else "",
               zorder=3, edgecolor='white')
        ax.text(x7[i] + 0.18, pv + 0.012, f"{pv:.4f}",
                ha='center', fontsize=8.5, fontweight='bold', color=pc)

for i, (rect, v) in enumerate(zip(bars_val, val_7)):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.012,
            f"{v:.4f}", ha='center', fontsize=8.5, color=BLUE if i == 0 else GREEN)

ax.annotate("Blend wins\n−0.101 private",
            xy=(x7[2] + 0.18, 0.7126),
            xytext=(x7[2] - 0.25, 0.66),
            fontsize=9, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3))

ax.set_xticks(x7)
ax.set_xticklabels(labels_7, fontsize=10)
ax.set_ylabel("WRMSSE (lower is better)", fontsize=10)
ax.set_ylim(0.45, 0.90)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)

# ---- Panel B: Day 10 — blend HURT ----
ax = axes[1]
ax.set_title("Day 10: Blending Hurt\n(per-store + global recursive)", fontsize=11, fontweight='bold', color=RED)

labels_10  = ["Global\nalone", "Per-store\nalone", "Blend\n(0.6×ps+0.4×g)"]
val_10     = [0.5422,  0.6140,  0.5737]
priv_10    = [None,    0.6410,  0.6430]
priv_c10   = [None,    TEAL,    TEAL]

x10 = np.arange(3)
bar_colors_10 = [BLUE, TEAL, TEAL]
bars_val10 = ax.bar(x10 - 0.18, val_10, 0.33, color=bar_colors_10, alpha=0.7,
                    label="Val WRMSSE", zorder=3, edgecolor='white')

for i, (pv, pc) in enumerate(zip(priv_10, priv_c10)):
    if pv is not None:
        ax.bar(x10[i] + 0.18, pv, 0.33, color=pc, alpha=1.0,
               label="Private LB" if i == 1 else "", zorder=3, edgecolor='white')
        ax.text(x10[i] + 0.18, pv + 0.012, f"{pv:.4f}",
                ha='center', fontsize=8.5, fontweight='bold', color=pc)

for rect, v, c in zip(bars_val10, val_10, bar_colors_10):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.012,
            f"{v:.4f}", ha='center', fontsize=8.5, color=c)

ax.annotate("Blend loses\n+0.002 private vs alone",
            xy=(x10[2] + 0.18, 0.6430),
            xytext=(x10[2] - 0.2, 0.68),
            fontsize=9, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.3))

ax.set_xticks(x10)
ax.set_xticklabels(labels_10, fontsize=10)
ax.set_ylabel("WRMSSE (lower is better)", fontsize=10)
ax.set_ylim(0.45, 0.90)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)

# Shared footnote
fig.text(0.5, -0.04,
         "Solid bars = Val WRMSSE (d_1914–1941, single-step)  |  Tall bars = Private LB (d_1942–1969, recursive)\n"
         "Day 7: global recursive alone = 0.8138 (above chart range, not shown as standalone)",
         ha='center', fontsize=8.5, color='#555555', style='italic')

plt.tight_layout()
path = os.path.join(CHARTS, "blend_dynamics.png")
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Chart 4: Hierarchical aggregation (unchanged) ─────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

labels  = ["Bottom-Up\nProphet", "Top-Down\nNational", "Top-Down\nState", "Top-Down\nDept", "Top-Down\nCategory"]
values  = [0.6638,               0.5580,               0.5740,            0.5565,            0.5555]
colors  = [ORANGE, BLUE, BLUE, BLUE, GREEN]

bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
for rect, v in zip(bars, values):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.005,
            f"{v:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

ax.axhline(0.6778, color=GREY, lw=1.2, ls='--', alpha=0.7)
ax.text(4.3, 0.682, "SN28 ref\n(0.6778)", fontsize=8.5, color=GREY, va='bottom')

ax.annotate("−0.108 vs BU",
            xy=(4, values[4]), xytext=(3.5, 0.52),
            fontsize=9.5, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

ax.set_ylabel("WRMSSE (lower is better)", fontsize=11)
ax.set_title("Top-Down vs Bottom-Up: Prophet on 1k Sample", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0.48, 0.75)
ax.grid(axis='y', alpha=0.3, zorder=0)

bu_patch   = mpatches.Patch(color=ORANGE, label="Bottom-Up")
td_patch   = mpatches.Patch(color=BLUE,   label="Top-Down (other)")
best_patch = mpatches.Patch(color=GREEN,  label="Top-Down Category (best)")
ax.legend(handles=[bu_patch, td_patch, best_patch], fontsize=9)

plt.tight_layout()
path = os.path.join(CHARTS, "hierarchical_aggregation.png")
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

print("\nAll 4 charts generated.")
