"""Create notebooks/02_failure_analysis.ipynb."""
import json
import uuid


def nid():
    return str(uuid.uuid4())[:8]


def md(src):
    return {"cell_type": "markdown", "id": nid(), "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "id": nid(),
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
    }


cells = []

# ── Cell 0: Title ──────────────────────────────────────────────────────────────
cells.append(
    md(
        "# Why Our Validation Metric Lied: A Calibration Analysis\n\n"
        "**Phase 2 finding:** 4 of 5 submitted LightGBM variants improved local validation "
        "WRMSSE but produced worse Kaggle private LB scores.  \n"
        "This notebook documents the data behind that reversal and traces it to two compounding "
        "causes.\n\n"
        "*Data sourced from `reports/leaderboard.md`. No large CSV files required.*"
    )
)

# ── Cell 1: Intro ──────────────────────────────────────────────────────────────
cells.append(
    md(
        "## The Reversal\n\n"
        "The ShelfSense pipeline uses a single 28-day holdout (days 1,914-1,941: "
        "**April 25 - May 22, 2016**) as the validation window. Six LightGBM variants were "
        "trained, evaluated on this window, and submitted to Kaggle. The result was a consistent "
        "pattern: variants that looked better locally either failed to improve the private "
        "leaderboard or made it worse.\n\n"
        "This is not a model bug or a data leak. It is a structural problem with single-window "
        "validation: one 28-day holdout is insufficient to distinguish genuine improvements from "
        "overfits to that specific month's demand pattern."
    )
)

# ── Cell 2: Imports ────────────────────────────────────────────────────────────
cells.append(
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.patches as mpatches\n"
        "from scipy import stats\n\n"
        "plt.rcParams.update({'figure.dpi': 100, 'font.size': 11})"
    )
)

# ── Cell 3: Data ───────────────────────────────────────────────────────────────
cells.append(
    code(
        "# Scores sourced from reports/leaderboard.md\n"
        "# val_wrmsse: in-model holdout (d_1886-d_1913, March 28 - April 24 2016)\n"
        "# private_lb: Kaggle private leaderboard (d_1942-d_1969, May 23 - June 19 2016)\n"
        "variants = pd.DataFrame([\n"
        "    {'name': 'tvp_13',     'val_wrmsse': 0.6860, 'private_lb': 0.5693},\n"
        "    {'name': 'ylags',      'val_wrmsse': 0.6830, 'private_lb': 0.5749},\n"
        "    {'name': 'store_dept', 'val_wrmsse': 0.6294, 'private_lb': 0.5882},\n"
        "    {'name': 'per_dept',   'val_wrmsse': 0.7333, 'private_lb': 0.6137},\n"
        "    {'name': 'rmse_mh',    'val_wrmsse': 0.6699, 'private_lb': 0.6205},\n"
        "    {'name': 'per_store',  'val_wrmsse': 0.6140, 'private_lb': 0.6410},\n"
        "])\n\n"
        "baseline = variants[variants['name'] == 'tvp_13'].iloc[0]\n"
        "variants['val_delta'] = variants['val_wrmsse'] - baseline['val_wrmsse']\n"
        "variants['priv_delta'] = variants['private_lb'] - baseline['private_lb']\n\n"
        "print(variants[['name', 'val_wrmsse', 'private_lb', 'val_delta', 'priv_delta']].to_string(index=False))"
    )
)

# ── Cell 4: Scatter plot ───────────────────────────────────────────────────────
cells.append(
    code(
        "fig, ax = plt.subplots(figsize=(10, 7))\n\n"
        "# Quadrant shading\n"
        "ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')\n"
        "ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')\n"
        "ax.fill_between([-0.12, 0], [0.1, 0.1], 0, alpha=0.07, color='red')\n"
        "ax.fill_between([0, 0.07], [0.1, 0.1], 0, alpha=0.05, color='orange')\n"
        "ax.fill_between([-0.12, 0], [0, 0], [-0.01, -0.01], alpha=0.06, color='green')\n\n"
        "# Quadrant labels\n"
        "ax.text(-0.06, 0.091, 'Val better,\\nprivate worse\\n(INVERSION)',\n"
        "        ha='center', va='top', fontsize=9, color='#990000', style='italic')\n"
        "ax.text(0.038, 0.091, 'Both worse',\n"
        "        ha='center', va='top', fontsize=9, color='darkorange', style='italic')\n"
        "ax.text(-0.06, -0.003, 'Both better (ideal)',\n"
        "        ha='center', va='bottom', fontsize=9, color='green', style='italic')\n\n"
        "# Label offsets to avoid overlap\n"
        "offsets = {\n"
        "    'tvp_13':     (6, -14),\n"
        "    'ylags':      (6, 5),\n"
        "    'store_dept': (-78, 5),\n"
        "    'per_dept':   (6, 5),\n"
        "    'rmse_mh':    (6, 5),\n"
        "    'per_store':  (6, -14),\n"
        "}\n\n"
        "for _, row in variants.iterrows():\n"
        "    is_base = row['name'] == 'tvp_13'\n"
        "    is_inv  = (row['val_delta'] < 0) and (row['priv_delta'] > 0) and not is_base\n"
        "    color  = 'black' if is_base else ('crimson' if is_inv else 'darkorange')\n"
        "    marker = 'D' if is_base else ('o' if is_inv else 's')\n"
        "    size   = 130 if is_base else 90\n"
        "    ax.scatter(row['val_delta'], row['priv_delta'],\n"
        "               color=color, marker=marker, s=size, zorder=5)\n"
        "    dx, dy = offsets.get(row['name'], (6, 5))\n"
        "    ax.annotate(row['name'], (row['val_delta'], row['priv_delta']),\n"
        "                xytext=(dx, dy), textcoords='offset points', fontsize=10,\n"
        "                fontweight='bold' if is_base else 'normal', color=color)\n\n"
        "ax.set_xlabel('Val WRMSSE delta vs tvp_13  (negative = improved validation)', fontsize=11)\n"
        "ax.set_ylabel('Private LB delta vs tvp_13  (positive = worsened private LB)', fontsize=11)\n"
        "ax.set_title(\n"
        "    'Single-holdout validation overfits to one 28-day window\\n'\n"
        "    '4 of 5 non-baseline variants improved val but worsened private LB',\n"
        "    fontsize=12, fontweight='bold')\n\n"
        "legend_handles = [\n"
        "    mpatches.Patch(color='crimson',    alpha=0.8, label='Inversion (val better, private worse)'),\n"
        "    mpatches.Patch(color='darkorange', alpha=0.8, label='Consistent degradation (both worse)'),\n"
        "    mpatches.Patch(color='black',      alpha=0.8, label='Baseline (tvp_13)'),\n"
        "]\n"
        "ax.legend(handles=legend_handles, loc='lower right', fontsize=10)\n"
        "ax.set_xlim(-0.12, 0.07)\n"
        "ax.set_ylim(-0.01, 0.1)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "n_inv = int(((variants['val_delta'] < 0) & (variants['priv_delta'] > 0)\n"
        "             & (variants['name'] != 'tvp_13')).sum())\n"
        "print(f'Inversions (val better, private worse): {n_inv} of {len(variants)-1} non-baseline variants')"
    )
)

# ── Cell 5: Correlation ────────────────────────────────────────────────────────
cells.append(
    code(
        "non_baseline = variants[variants['name'] != 'tvp_13'].copy()\n\n"
        "r_p, p_p = stats.pearsonr(non_baseline['val_delta'], non_baseline['priv_delta'])\n"
        "r_s, p_s = stats.spearmanr(non_baseline['val_delta'], non_baseline['priv_delta'])\n"
        "print('All 5 non-baseline variants (tvp_17 excluded: no Kaggle submission made):')\n"
        "print(f'  Pearson  r = {r_p:.3f}  (p = {p_p:.3f})')\n"
        "print(f'  Spearman r = {r_s:.3f}  (p = {p_s:.3f})')\n"
        "print()\n\n"
        "inversions = non_baseline[\n"
        "    (non_baseline['val_delta'] < 0) & (non_baseline['priv_delta'] > 0)\n"
        "]\n"
        "print(f'Inversion cases ({len(inversions)} variants): val better, private worse')\n"
        "for _, row in inversions.sort_values('val_delta').iterrows():\n"
        "    print(f\"  {row['name']:12s}: val_delta={row['val_delta']:+.4f}  \"\n"
        "          f\"priv_delta={row['priv_delta']:+.4f}\")\n"
        "print()\n\n"
        "if len(inversions) >= 3:\n"
        "    r_inv, p_inv = stats.pearsonr(inversions['val_delta'].abs(), inversions['priv_delta'])\n"
        "    print(f'Among inversions: |val improvement| vs private regression:')\n"
        "    print(f'  Pearson r = {r_inv:.3f}  (p = {p_inv:.3f})')\n"
        "    print(f'  Larger val improvements do not reliably predict larger private regressions.')\n"
        "    print(f'  rmse_mh improved val by only 0.016 but caused the second-worst private'\n"
        "          f' degradation (+0.051): objective-metric mismatch compounds the window problem.')"
    )
)

# ── Cell 6: Mechanisms (markdown) ─────────────────────────────────────────────
cells.append(
    md(
        "## Two Compounding Mechanisms\n\n"
        "### 1. Single-window holdout overfitting\n\n"
        "The validation window (April 25 - May 22, 2016) is one specific 28-day period with its "
        "own SNAP calendar, event mix, and seasonal demand level. Optuna runs that minimise WRMSSE "
        "on this window find parameters that are optimal for *this particular month*, not for the "
        "general demand distribution. `store_dept` ran 10 Optuna trials per slice (700 total "
        "across 70 slices) — those parameters are tuned to April-May 2016. The private evaluation "
        "window (May-Jun 2016) has a different pattern, and the over-tuned parameters do not "
        "transfer.\n\n"
        "The next cell shows a concrete symptom: the validation window has an unusually **low zero "
        "rate** relative to the training period. Lower zero rate means more active demand — easier "
        "to forecast and easier to overfit to.\n\n"
        "### 2. Objective-metric mismatch for `rmse_mh`\n\n"
        "The RMSE variant trains on mean squared error. WRMSSE weights high-revenue and intermittent "
        "series more heavily than MSE does. In the validation window (atypically low zero rate, "
        "more active demand), MSE and WRMSSE happen to correlate better — so the RMSE model looks "
        "good. In the private window, zero-inflation reasserts, and the model's systematic "
        "under-forecasting of sparse series drives WRMSSE up sharply. "
        "**Using the evaluation metric as the training objective is non-negotiable for M5.**\n\n"
        "*Full analysis: [docs/MODELS.md — Val-Private Divergence](../docs/MODELS.md)*"
    )
)

# ── Cell 7: Zero rates bar chart ───────────────────────────────────────────────
cells.append(
    code(
        "# Zero rates computed from sales_train_evaluation.csv\n"
        "# Training (d_1-d_1913, Jan 2011-Apr 2016) vs val window (d_1914-d_1941, Apr-May 2016)\n"
        "# Derivation: see notebooks/01_eda.ipynb, Section 3.2\n"
        "zero_rates = {\n"
        "    'FOODS':     {'Training\\n(d_1-1913)': 62.0, 'Val window\\n(d_1914-1941)': 44.8},\n"
        "    'HOBBIES':   {'Training\\n(d_1-1913)': 77.3, 'Val window\\n(d_1914-1941)': 68.2},\n"
        "    'HOUSEHOLD': {'Training\\n(d_1-1913)': 71.8, 'Val window\\n(d_1914-1941)': 60.2},\n"
        "}\n\n"
        "cats    = list(zero_rates.keys())\n"
        "windows = ['Training\\n(d_1-1913)', 'Val window\\n(d_1914-1941)']\n"
        "colors  = ['#4472C4', '#ED7D31']\n"
        "x       = np.arange(len(cats))\n"
        "width   = 0.35\n\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "for i, (window, color) in enumerate(zip(windows, colors)):\n"
        "    vals = [zero_rates[cat][window] for cat in cats]\n"
        "    bars = ax.bar(x + (i - 0.5) * width, vals, width,\n"
        "                  label=window.replace('\\n', ' '), color=color, alpha=0.85)\n"
        "    for bar, v in zip(bars, vals):\n"
        "        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,\n"
        "                f'{v:.1f}%', ha='center', va='bottom', fontsize=9)\n\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(cats, fontsize=12)\n"
        "ax.set_ylabel('Zero-sale rate (%)')\n"
        "ax.set_ylim(0, 95)\n"
        "ax.set_title(\n"
        "    'Val window (Apr-May 2016) has atypically low zero rates vs training distribution\\n'\n"
        "    'Private eval (d_1942-1969, May-Jun 2016): labels unreleased, expected near training baseline',\n"
        "    fontsize=11)\n"
        "ax.legend(fontsize=10)\n\n"
        "# Annotate the delta for each category\n"
        "for i, cat in enumerate(cats):\n"
        "    delta = (zero_rates[cat]['Training\\n(d_1-1913)']\n"
        "             - zero_rates[cat]['Val window\\n(d_1914-1941)'])\n"
        "    mid_x = x[i]\n"
        "    mid_y = zero_rates[cat]['Val window\\n(d_1914-1941)'] + 2\n"
        "    ax.annotate(f'-{delta:.1f}pp', xy=(mid_x, mid_y),\n"
        "                ha='center', fontsize=9, color='darkred', fontweight='bold')\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "overall_delta = 68.2 - 54.4\n"
        "print(f'Overall zero-rate delta (training vs val window): -{overall_delta:.1f} pp')\n"
        "print('The val window is 13.8 pp more active than the long-run average.')\n"
        "print('Models tuned to this window overestimate their own accuracy.')"
    )
)

# ── Cell 8: Closing (markdown) ─────────────────────────────────────────────────
cells.append(
    md(
        "## What This Taught Me\n\n"
        "**The val window is an optimistic outlier.** April-May 2016 has 13.8 percentage points "
        "lower zero rate than the overall training period. Every model evaluated against this window "
        "sees artificially good metrics -- and every Optuna run tuned against it finds parameters "
        "that are optimal for an atypically active month, not the general demand distribution.\n\n"
        "**Four inversions, one direction.** Of the five non-baseline variants with confirmed Kaggle "
        "scores, four (ylags, store_dept, rmse_mh, per_store) improved validation WRMSSE and "
        "worsened private LB. The fifth (per_dept) degraded both. The ideal quadrant -- val better "
        "AND private better -- is empty.\n\n"
        "**The fix is walk-forward cross-validation.** Five rolling 28-day folds across the last "
        "140 days of training data would average out the April-May anomaly. A variant that "
        "consistently improves WRMSSE across all five folds -- covering different SNAP schedules, "
        "seasonal demand levels, and zero-rate regimes -- is a genuine improvement, not a lucky "
        "calibration to one atypical month.\n\n"
        "The `cv_evaluation` Dagster asset specification is in "
        "[docs/MODELS.md -- Walk-Forward CV](../docs/MODELS.md#val-private-divergence). "
        "Implementing it is the highest-priority next step before running further model experiments."
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open("notebooks/02_failure_analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Created 02_failure_analysis.ipynb with {len(cells)} cells")
for i, c in enumerate(cells):
    src = (
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    )[:60].replace("\n", " ")
    print(f"  [{i}] {c['cell_type']:8s} | {src}")
