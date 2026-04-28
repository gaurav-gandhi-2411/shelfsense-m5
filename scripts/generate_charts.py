"""
Day 10: Generate portfolio charts for README.

Charts produced:
  reports/charts/leaderboard_progression.png  — WRMSSE improvement across methods
  reports/charts/per_category_comparison.png  — FOODS/HOUSEHOLD/HOBBIES across models
  reports/charts/hierarchical_aggregation.png — Top-down vs bottom-up Prophet
  reports/charts/public_vs_private_lb.png     — Public vs private Kaggle LB scatter
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS = os.path.join(PROJ_ROOT, "reports", "charts")
os.makedirs(CHARTS, exist_ok=True)

BLUE   = "#2C6FAC"
ORANGE = "#E07B39"
GREEN  = "#3A8C5C"
RED    = "#C0392B"
GREY   = "#888888"
LIGHT  = "#D0E4F5"

# ── Chart 1: Leaderboard progression ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))

# Full-catalogue methods (solid, dark)
full_labels  = ["SN28\nbaseline", "LightGBM\nRMSE", "LightGBM\nTweedie", "LightGBM\nOptuna", "Blend\n(Day 7)"]
full_values  = [0.8377,           0.5651,            0.5442,              0.5422,             0.5545]
full_x       = [0, 4, 5, 6, 7]

# Sample-based methods (hatched, lighter)
sample_labels = ["ETS\n(1k sample)", "ARIMA\n(1k sample)", "Prophet\n(1k sample)", "TD-Prophet\n(1k sample)"]
sample_values = [0.6541, 0.7493, 0.6638, 0.5555]
sample_x      = [1, 2, 3, 3.7]

all_x      = full_x + sample_x
all_vals   = full_values + sample_values
all_labels = full_labels + sample_labels
all_colors = [BLUE]*5 + [ORANGE]*4
all_hatch  = ['']*5 + ['////']*4

ax.axhline(0.8377, color=GREY, lw=1, ls='--', alpha=0.5)

bars = ax.bar(
    full_x, full_values,
    color=BLUE, width=0.6, zorder=3, label="Full 30,490 series"
)
bars2 = ax.bar(
    sample_x, sample_values,
    color=ORANGE, width=0.6, hatch='////', alpha=0.85, zorder=3, label="1k-sample (not directly comparable)"
)

# Value labels
for rect, v in zip(bars, full_values):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.008, f"{v:.4f}",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=BLUE)
for rect, v in zip(bars2, sample_values):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.008, f"{v:.4f}*",
            ha='center', va='bottom', fontsize=8.5, color=ORANGE)

ax.set_xticks(full_x + sample_x)
ax.set_xticklabels(full_labels + sample_labels, fontsize=9)
ax.set_ylabel("WRMSSE (lower is better)", fontsize=11)
ax.set_title("ShelfSense-M5: Model Progression (WRMSSE)", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.text(6, 0.52, "Best public LB\n0.5422", ha='center', fontsize=8, color=BLUE, style='italic')
ax.text(7, 0.57, "Best private LB\n0.7126", ha='center', fontsize=8, color=BLUE, style='italic')
ax.text(1.35, 0.69, "* sample scores\nnot directly comparable\nto full-catalogue", fontsize=7.5,
        color=ORANGE, ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor=ORANGE, alpha=0.7))

plt.tight_layout()
path = os.path.join(CHARTS, "leaderboard_progression.png")
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Chart 2: Per-category WRMSSE comparison ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

categories = ["FOODS", "HOUSEHOLD", "HOBBIES"]
ets_scores  = [0.5616, 1.7023, 3.2663]
lgbm_scores = [0.5204, 0.5905, 0.6112]

x     = np.arange(len(categories))
width = 0.32

b1 = ax.bar(x - width/2, ets_scores,  width, color=ORANGE, label="ETS (1k sample)", zorder=3)
b2 = ax.bar(x + width/2, lgbm_scores, width, color=BLUE,   label="LightGBM Tweedie (full)", zorder=3)

for rect, v in zip(b1, ets_scores):
    label = f"{v:.4f}" if v < 2.5 else f"{v:.2f}\n(zero-fallback)"
    ax.text(rect.get_x() + rect.get_width()/2, min(v, 2.55) + 0.04,
            label, ha='center', va='bottom', fontsize=8.5, color=ORANGE, fontweight='bold')
for rect, v in zip(b2, lgbm_scores):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.04,
            f"{v:.4f}", ha='center', va='bottom', fontsize=8.5, color=BLUE, fontweight='bold')

ax.annotate("5× improvement\n(3.27 → 0.61)",
            xy=(x[2] + width/2, lgbm_scores[2]),
            xytext=(x[2] + width/2 + 0.45, 1.4),
            fontsize=9.5, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel("WRMSSE (lower is better)", fontsize=11)
ax.set_title("WRMSSE by Category: ETS vs LightGBM", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, 3.0)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, zorder=0)

# Mark ETS HOBBIES bar as clipped
ax.annotate("3.27 (clipped)", xy=(x[2] - width/2, 2.55),
            fontsize=8, ha='center', va='bottom', color=ORANGE, style='italic')
ax.set_ylim(0, 2.8)

plt.tight_layout()
path = os.path.join(CHARTS, "per_category_comparison.png")
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Chart 3: Hierarchical aggregation (Top-Down vs Bottom-Up) ─────────────────
fig, ax = plt.subplots(figsize=(8, 5))

labels  = ["Bottom-Up\nProphet", "Top-Down\nNational", "Top-Down\nState", "Top-Down\nDept", "Top-Down\nCategory"]
values  = [0.6638,               0.5580,               0.5740,            0.5565,            0.5555]
colors  = [ORANGE, BLUE, BLUE, BLUE, GREEN]

bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
for rect, v, lbl in zip(bars, values, labels):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.005,
            f"{v:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='#333333')

ax.axhline(0.6778, color=GREY, lw=1.2, ls='--', alpha=0.7)
ax.text(4.3, 0.682, "SN28 ref\n(0.6778)", fontsize=8.5, color=GREY, va='bottom')

ax.annotate("-0.108 vs BU",
            xy=(4, values[4]), xytext=(3.5, 0.52),
            fontsize=9.5, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

ax.set_ylabel("WRMSSE (lower is better)", fontsize=11)
ax.set_title("Top-Down vs Bottom-Up: Prophet on 1k Sample", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0.48, 0.75)
ax.grid(axis='y', alpha=0.3, zorder=0)

bu_patch  = mpatches.Patch(color=ORANGE, label="Bottom-Up")
td_patch  = mpatches.Patch(color=BLUE,   label="Top-Down (other)")
best_patch = mpatches.Patch(color=GREEN, label="Top-Down Category (best)")
ax.legend(handles=[bu_patch, td_patch, best_patch], fontsize=9)

plt.tight_layout()
path = os.path.join(CHARTS, "hierarchical_aggregation.png")
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Chart 4: Public vs Private LB scatter ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5))

points = [
    # (public, private, label, color, marker)
    (0.8377, 0.8956, "SN28 baseline",              GREY,   "o"),
    (0.8377, 0.8698, "ETS (1k+SN28)",              ORANGE, "s"),
    (0.8377, 0.8582, "ARIMA (1k+SN28)",            ORANGE, "^"),
    (0.8377, 0.8731, "Prophet (1k+SN28)",          ORANGE, "D"),
    (0.5422, 0.8956, "LightGBM Day 6\n(SN28 eval)",BLUE,   "o"),
    (0.5422, 0.8138, "LightGBM\nglobal recursive", BLUE,   "s"),
    (0.5545, 0.7126, "Blend\n(best private)",      GREEN,  "*"),
]

for pub, priv, lbl, col, mrk in points:
    ms = 180 if mrk == "*" else 90
    ax.scatter(pub, priv, c=col, marker=mrk, s=ms, zorder=5, edgecolors='white', linewidths=0.8)
    offset_x = 0.003 if pub < 0.6 else -0.003
    offset_y = 0.006 if priv < 0.87 else -0.012
    align = 'left' if pub < 0.6 else 'right'
    ax.annotate(lbl, (pub, priv), xytext=(pub + offset_x, priv + offset_y),
                fontsize=8, color=col, ha=align, fontweight='bold' if col == GREEN else 'normal')

# Diagonal reference: public == private
diag = np.linspace(0.5, 0.92, 100)
ax.plot(diag, diag, color=GREY, lw=1, ls=':', alpha=0.5, label="public = private")

# Blend annotation
ax.annotate("Worse public\nBetter private", xy=(0.5545, 0.7126), xytext=(0.62, 0.69),
            fontsize=9, color=GREEN, style='italic',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))

ax.set_xlabel("Kaggle Public LB Score (WRMSSE)", fontsize=11)
ax.set_ylabel("Kaggle Private LB Score (WRMSSE)", fontsize=11)
ax.set_title("Public vs Private Leaderboard Score\n(lower is better)", fontsize=13,
             fontweight='bold', pad=10)
ax.set_xlim(0.48, 0.90)
ax.set_ylim(0.65, 0.93)
ax.grid(alpha=0.25, zorder=0)

handles = [
    mpatches.Patch(color=GREY,   label="Baseline"),
    mpatches.Patch(color=ORANGE, label="Classical (1k-sample)"),
    mpatches.Patch(color=BLUE,   label="LightGBM"),
    mpatches.Patch(color=GREEN,  label="Blend (best)"),
]
ax.legend(handles=handles, fontsize=9, loc='lower right')

plt.tight_layout()
path = os.path.join(CHARTS, "public_vs_private_lb.png")
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

print("\nAll 4 charts generated.")
