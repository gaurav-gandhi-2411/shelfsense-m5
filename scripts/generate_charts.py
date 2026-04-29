"""
Generate portfolio charts for README.

Charts produced:
  reports/charts/leaderboard_progression.png  — private LB over time with phase annotations
  reports/charts/per_category_journey.png     — FOODS/HOUSEHOLD/HOBBIES across key methods
  reports/charts/blend_dynamics.png           — when blending helped (Day 7) vs hurt (Day 10)
  reports/charts/hierarchical_aggregation.png — top-down vs bottom-up Prophet
"""
import os
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
PURPLE = "#7B3FA0"
GREY   = "#888888"
TEAL   = "#1A7F7A"
LGREY  = "#CCCCCC"


# ── helpers ───────────────────────────────────────────────────────────────────

def value_label(ax, rect, v, fmt="{:.4f}", color="#333333", size=9.5, pad=0.012):
    ax.text(rect.get_x() + rect.get_width() / 2, v + pad,
            fmt.format(v), ha="center", va="bottom",
            fontsize=size, fontweight="bold", color=color)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Private LB Progression (chronological)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

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
    "Per-store\nblend\nDay 10",
]
private_lb = [0.8956, 0.8698, 0.8582, 0.8956, 0.8138, 0.7126, 0.7126, 0.6095, 0.5854, 0.6410, 0.6430]
x = np.arange(len(methods))

bar_colors = [GREY, ORANGE, ORANGE, BLUE, BLUE, GREEN, GREEN, PURPLE, PURPLE, TEAL, TEAL]
bars = ax.bar(x, private_lb, color=bar_colors, width=0.62, zorder=3,
              edgecolor="white", linewidth=0.6)

for rect, v in zip(bars, private_lb):
    value_label(ax, rect, v, pad=0.014, size=9.5)

# Best-to-date step line
best_so_far, cur = [], 999.0
for v in private_lb:
    cur = min(cur, v)
    best_so_far.append(cur)
ax.step(np.append(x - 0.31, x[-1] + 0.31), np.append(best_so_far, best_so_far[-1]),
        where="post", color=RED, lw=2.0, ls="--", zorder=6, label="Best private LB to date")
ax.plot(x[-1] + 0.31, best_so_far[-1], "D", color=RED, ms=7, zorder=7)

# Phase separator lines (drawn before phase labels so labels sit on top)
PHASE_CUTS = [2.5, 6.5, 8.5]
for xc in PHASE_CUTS:
    ax.axvline(xc, color=LGREY, lw=1.4, ls="--", zorder=2)

# Phase labels — positioned in top headroom, well above all bars
PHASE_REGIONS = [
    (1.0, "Classical\n(1k sample)"),
    (4.5, "LightGBM\nglobal"),
    (7.5, "Multi-horizon\n(Day 9)"),
    (9.5, "Per-store\n(Day 10)"),
]
for px, label in PHASE_REGIONS:
    ax.text(px, 1.075, label, ha="center", va="bottom", fontsize=10,
            color="#444444", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor=LGREY, alpha=0.9))

# ── Inflection annotations: xytext in top headroom, straight-down arrows ──

# 1. Recursive eval fixed (bar 3: LightGBM Day 6 was SN28 placeholder)
ax.annotate(
    "Eval rows filled with SN28\nuntil recursive forecast added",
    xy=(3, 0.8956 + 0.015), xytext=(3, 1.040),
    ha="center", va="bottom", fontsize=9, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF4FB", edgecolor=BLUE, alpha=0.9),
)

# 2. LightGBM cross-series learning (bar 4: first real private-LB improvement)
ax.annotate(
    "Cross-series learning\n−35% vs SN28 baseline",
    xy=(4, 0.8138 + 0.015), xytext=(4, 1.040),
    ha="center", va="bottom", fontsize=9, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF4FB", edgecolor=BLUE, alpha=0.9),
)

# 3. Multi-horizon best result (bar 8)
ax.annotate(
    "Direct 28-step prediction\neliminates compounding  ★ best",
    xy=(8, 0.5854 + 0.015), xytext=(8, 0.750),
    ha="center", va="bottom", fontsize=9, color=PURPLE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0E8F8", edgecolor=PURPLE, alpha=0.9),
)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9.5)
ax.set_ylabel("Private LB WRMSSE (lower is better)", fontsize=11)
ax.set_title("ShelfSense-M5: Kaggle Private LB Progression (Chronological)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_ylim(0.44, 1.13)
ax.grid(axis="y", alpha=0.25, zorder=0)

handles = [
    mpatches.Patch(color=GREY,   label="Baseline"),
    mpatches.Patch(color=ORANGE, label="Classical (1k fill)"),
    mpatches.Patch(color=BLUE,   label="LightGBM global"),
    mpatches.Patch(color=GREEN,  label="Blend — per-category"),
    mpatches.Patch(color=PURPLE, label="Multi-horizon (Day 9)"),
    mpatches.Patch(color=TEAL,   label="Per-store (Day 10)"),
    plt.Line2D([0], [0], color=RED, lw=2, ls="--", label="Best private LB to date"),
]
ax.legend(handles=handles, fontsize=9, loc="lower left", ncol=4,
          bbox_to_anchor=(0.0, -0.22), frameon=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
path = os.path.join(CHARTS, "leaderboard_progression.png")
plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Per-category journey
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

categories   = ["FOODS", "HOUSEHOLD", "HOBBIES"]
sn28_scores  = [0.6400,  1.1580,      1.5949]
ets_scores   = [0.5616,  1.7023,      3.2663]
lgbm_scores  = [0.5204,  0.5905,      0.6112]

x     = np.arange(len(categories))
width = 0.24
CLIP  = 2.80   # ETS HOBBIES (3.27) clipped here

b1 = ax.bar(x - width, sn28_scores, width, color=GREY,   label="SN28 baseline",          zorder=3)
b2 = ax.bar(x,         ets_scores,  width, color=ORANGE, label="ETS (1k sample)",         zorder=3,
            alpha=0.85, hatch="////")
b3 = ax.bar(x + width, lgbm_scores, width, color=BLUE,   label="LightGBM Tweedie (full)", zorder=3)

# SN28 labels
for rect, v in zip(b1, sn28_scores):
    value_label(ax, rect, v, color=GREY, size=9.5)

# ETS labels — special handling for HOBBIES (clipped)
for i, (rect, v) in enumerate(zip(b2, ets_scores)):
    display_v = min(v, CLIP)
    if v > CLIP:
        ax.text(rect.get_x() + rect.get_width() / 2, CLIP + 0.06,
                f"3.27 ↑", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=ORANGE)
    else:
        value_label(ax, rect, v, color=ORANGE, size=9.5)

# LightGBM labels
for rect, v in zip(b3, lgbm_scores):
    value_label(ax, rect, v, color=BLUE, size=9.5)

# "Zero-fallback" annotation — text placed above chart, arrow pointing down to ETS HOBBIES bar top
ets_hobbies_bar = b2[2]
ets_hobbies_x   = ets_hobbies_bar.get_x() + ets_hobbies_bar.get_width() / 2
ax.annotate(
    "Zero-forecast fallback\n(390/1k sparse series)",
    xy=(ets_hobbies_x, CLIP + 0.06), xytext=(ets_hobbies_x - 0.55, CLIP + 0.40),
    ha="center", va="bottom", fontsize=9.5, color=ORANGE, fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor=ORANGE, alpha=0.85),
)

# "5× improvement" — text to the right of HOBBIES LightGBM bar, pointing left to bar top
lgbm_hobbies_bar = b3[2]
lgbm_hobbies_x   = lgbm_hobbies_bar.get_x() + lgbm_hobbies_bar.get_width() / 2
ax.annotate(
    "5× improvement\n3.27 → 0.61\n(cross-series signal)",
    xy=(lgbm_hobbies_x, lgbm_scores[2] + 0.02), xytext=(lgbm_hobbies_x + 0.55, 1.00),
    ha="left", va="bottom", fontsize=9.5, color=RED, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor=RED, alpha=0.9),
)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylabel("WRMSSE (lower is better)", fontsize=11)
ax.set_title("Per-Category WRMSSE: Baseline → Classical → LightGBM",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 3.35)
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", alpha=0.25, zorder=0)

# Clip marker: dashed horizontal line at CLIP with note
ax.axhline(CLIP, color=ORANGE, lw=1.0, ls=":", alpha=0.5)
ax.text(0.02, CLIP + 0.04, f"chart clipped at {CLIP}", color=ORANGE,
        fontsize=8, fontstyle="italic", transform=ax.get_yaxis_transform())

plt.tight_layout()
path = os.path.join(CHARTS, "per_category_journey.png")
plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Blend dynamics — Day 7 helped, Day 10 hurt
# ─────────────────────────────────────────────────────────────────────────────
#
# Each panel shows private LB bars only (the main comparison).
# Val WRMSSE shown as text below x-axis labels for context.
# Keeps annotations in clear whitespace — no text on bar bodies.
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
fig.patch.set_facecolor("white")
fig.suptitle("Ensemble Diversity: When Blending Helps vs Hurts",
             fontsize=14, fontweight="bold", y=1.01)

# ── Panel A: Day 7 — blend HELPED ────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("white")
ax.set_title("Day 7: Blending Helped\n(per-category + global)", fontsize=12,
             fontweight="bold", color=GREEN, pad=10)

labels_7  = ["Global recursive\nval WRMSSE: 0.5422", "Blend\n(0.6×per-cat + 0.4×global)\nval WRMSSE: 0.5545"]
private_7 = [0.8138, 0.7126]
x7 = np.arange(2)

bars7 = ax.bar(x7, private_7, color=[BLUE, GREEN], width=0.45, zorder=3,
               edgecolor="white", linewidth=0.6)

for rect, v in zip(bars7, private_7):
    value_label(ax, rect, v, size=11, pad=0.015)

# Improvement annotation — placed in clear space LEFT of global bar, pointing right
ax.annotate(
    "Blend −0.101\non private LB",
    xy=(1, 0.7126 + 0.015),   # top of blend bar
    xytext=(0.5, 0.885),       # clear space above and between
    ha="center", va="bottom", fontsize=10, color=GREEN, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor=GREEN, alpha=0.9),
)

ax.set_xticks(x7)
ax.set_xticklabels(labels_7, fontsize=11)
ax.set_ylabel("Private LB WRMSSE (lower is better)", fontsize=11)
ax.set_ylim(0.60, 0.97)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.legend(handles=[
    mpatches.Patch(color=BLUE,  label="Global recursive"),
    mpatches.Patch(color=GREEN, label="Blend"),
], fontsize=9, loc="lower right")

# ── Panel B: Day 10 — blend HURT ─────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor("white")
ax.set_title("Day 10: Blending Hurt\n(per-store + global recursive)", fontsize=12,
             fontweight="bold", color=RED, pad=10)

labels_10  = ["Per-store alone\nval WRMSSE: 0.6140", "Blend\n(0.6×per-store + 0.4×global)\nval WRMSSE: 0.5737"]
private_10 = [0.6410, 0.6430]
x10 = np.arange(2)

TEAL_LIGHT = "#5AADA8"
bar10_colors = [TEAL, TEAL_LIGHT]
bars10 = ax.bar(x10, private_10, color=bar10_colors, width=0.45, zorder=3,
                edgecolor="white", linewidth=0.6)

for rect, v in zip(bars10, private_10):
    value_label(ax, rect, v, size=11, pad=0.003)

# Degradation annotation — placed above both bars, pointing down to blend bar top
ax.annotate(
    "Blend +0.002\n(global recursive\nadds noise, not signal)",
    xy=(1, 0.6430 + 0.003),   # top of blend bar
    xytext=(0.5, 0.750),       # clear space above bars
    ha="center", va="bottom", fontsize=10, color=RED, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor=RED, alpha=0.9),
)

ax.set_xticks(x10)
ax.set_xticklabels(labels_10, fontsize=11)
ax.set_ylabel("Private LB WRMSSE (lower is better)", fontsize=11)
ax.set_ylim(0.60, 0.97)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.legend(handles=[
    mpatches.Patch(color=TEAL,       label="Per-store alone"),
    mpatches.Patch(color=TEAL_LIGHT, label="Blend"),
], fontsize=9, loc="lower right")

# Shared insight box below both panels
fig.text(
    0.5, -0.06,
    "Day 7: global and per-category were comparably imperfect → diversity gain on private LB\n"
    "Day 10: per-store already outperformed global recursive → blending in the weaker component added noise",
    ha="center", fontsize=9.5, color="#444444", fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8", edgecolor=LGREY, alpha=0.9),
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
path = os.path.join(CHARTS, "blend_dynamics.png")
plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Hierarchical aggregation
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

labels  = ["Bottom-Up\nProphet", "Top-Down\nNational", "Top-Down\nState",
           "Top-Down\nDept", "Top-Down\nCategory"]
values  = [0.6638, 0.5580, 0.5740, 0.5565, 0.5555]
colors  = [ORANGE, BLUE, BLUE, BLUE, GREEN]
x5      = np.arange(len(labels))

bars = ax.bar(x5, values, color=colors, width=0.55, zorder=3)
for rect, v in zip(bars, values):
    value_label(ax, rect, v, size=11, pad=0.007)

# SN28 reference line
ax.axhline(0.6778, color=GREY, lw=1.2, ls="--", alpha=0.7, zorder=2)
ax.text(4.35, 0.6778 + 0.004, "SN28 ref (0.6778)", fontsize=9.5, color=GREY, va="bottom")

# "-0.108 vs Bottom-Up" annotation: text in upper-left area, arrow pointing to category bar top
ax.annotate(
    "−0.108 vs Bottom-Up\n(category aggregation\ncancels sparse-series noise)",
    xy=(4, 0.5555 + 0.007),    # top of Top-Down Category bar
    xytext=(1.5, 0.735),        # clear headroom — upper-center of chart
    ha="center", va="bottom", fontsize=9.5, color=GREEN, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5,
                    connectionstyle="arc3,rad=-0.15"),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor=GREEN, alpha=0.9),
)

ax.set_xticks(x5)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("WRMSSE (lower is better)", fontsize=11)
ax.set_title("Top-Down vs Bottom-Up: Prophet on 1k Sample", fontsize=13,
             fontweight="bold", pad=12)
ax.set_ylim(0.47, 0.78)
ax.grid(axis="y", alpha=0.25, zorder=0)

ax.legend(handles=[
    mpatches.Patch(color=ORANGE, label="Bottom-Up"),
    mpatches.Patch(color=BLUE,   label="Top-Down (other levels)"),
    mpatches.Patch(color=GREEN,  label="Top-Down Category (best)"),
], fontsize=10, loc="upper right")

plt.tight_layout()
path = os.path.join(CHARTS, "hierarchical_aggregation.png")
plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {path}")

print("\nAll 4 charts generated.")
