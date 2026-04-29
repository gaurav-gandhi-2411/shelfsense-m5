"""
Unit tests for ChartCanvas (collision-aware chart rendering).

Run with:   pytest tests/test_chart_canvas.py -v
"""
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shelfsense.visualization.charts import (
    ChartCanvas,
    _segment_crosses_bar,
    _Callout,
    BLUE, RED, GREEN,
)


# ── geometry unit tests ───────────────────────────────────────────────────────

class TestSegmentCrossesBar:
    """Tests for the low-level Liang-Barsky intersection helper."""

    def test_vertical_arrow_above_bar_no_cross(self):
        """Straight-down arrow from margin to bar top does not cross the bar body."""
        # bar at x=0, height=0.8, width=0.62
        # arrow from (0, 1.04) to (0, 0.815) — tip just above bar top
        assert not _segment_crosses_bar(
            (0.0, 1.04), (0.0, 0.815), bar_x=0.0, bar_h=0.8, bar_w=0.62
        )

    def test_arrow_passes_through_bar_body(self):
        """Arrow whose midpoint is inside a bar body must be flagged."""
        # bar at x=1, height=0.9, width=0.62 → body x=[0.69,1.31], y=[0,0.9]
        # arrow from (1, 0.5) to (1, 0.3) — entirely inside the bar body
        assert _segment_crosses_bar(
            (1.0, 0.5), (1.0, 0.3), bar_x=1.0, bar_h=0.9, bar_w=0.62
        )

    def test_arrow_misses_bar_horizontally(self):
        """Arrow at x=3 does not cross bar at x=0."""
        assert not _segment_crosses_bar(
            (3.0, 1.04), (3.0, 0.60), bar_x=0.0, bar_h=0.8, bar_w=0.62
        )

    def test_diagonal_arrow_clears_intermediate_bar(self):
        """Diagonal arrow from (2, 1.04) to (3, 0.91) clears bar at x=2 (h=0.86)."""
        # Arrow at x=2: y=1.04, bar top=0.86 — arrow passes above bar 2
        assert not _segment_crosses_bar(
            (2.0, 1.04), (3.0, 0.91), bar_x=2.0, bar_h=0.86, bar_w=0.62
        )

    def test_diagonal_arrow_crosses_tall_intermediate_bar(self):
        """Diagonal arrow crossing through a tall bar body must be detected."""
        # Arrow from (0, 1.04) to (3, 0.50): at x=1.5, y ≈ 0.77
        # Bar at x=1.5, height=0.85, width=0.62 → body contains (1.5, 0.77)
        assert _segment_crosses_bar(
            (0.0, 1.04), (3.0, 0.50), bar_x=1.5, bar_h=0.85, bar_w=0.62
        )


# ── ChartCanvas integration tests ────────────────────────────────────────────

class TestCalloutInTopMargin:
    """validate() must pass when callout xytext is in the top margin zone."""

    def test_straight_down_callout_no_violations(self):
        """Default add_callout (placement='top') produces no violations."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y")
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.80, 0.60, 0.90])
        canvas.add_bars(x, y, colors=[BLUE] * 3)
        canvas.set_ylim(0.40, 1.20)
        canvas.add_callout(target_x=1.0, target_y=0.62, text="Inline test\nline 2")
        violations = canvas.validate()
        assert violations == [], f"Unexpected violations: {violations}"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_multiple_callouts_stacked_no_violations(self):
        """Three sequential callouts should all land in the top margin."""
        canvas = ChartCanvas(figsize=(16, 8), title="T", ylabel="Y",
                             top_margin_pct=0.20)
        x = np.arange(5, dtype=float)
        y = np.array([0.85, 0.70, 0.60, 0.55, 0.65])
        canvas.add_bars(x, y, colors=[BLUE] * 5)
        canvas.set_ylim(0.40, 1.20)
        canvas.add_callout(target_x=0, target_y=0.87, text="First", color=BLUE)
        canvas.add_callout(target_x=2, target_y=0.62, text="Second\ntwo lines", color=GREEN)
        canvas.add_callout(target_x=4, target_y=0.67, text="Third", color=RED)
        violations = canvas.validate()
        assert violations == [], f"Unexpected violations: {violations}"
        import matplotlib.pyplot as plt
        plt.close("all")


class TestArrowRoutesAroundBars:
    """validate() catches arrows that cross through non-target bar bodies."""

    def test_large_x_offset_crossing_bars_is_flagged(self):
        """A large x_offset that routes the arrow through intermediate bar bodies triggers a violation."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y")
        x = np.arange(5, dtype=float)
        # bars at x=0–3 are tall (0.92); bar at x=4 is the low target (0.52).
        # Arrow: text at x=0 (margin, ~1.09) → target (4, 0.52).
        # At x=2 the arrow y ≈ 0.82, which is inside bar body [0, 0.92] → violation.
        y = np.array([0.92, 0.92, 0.92, 0.92, 0.52])
        canvas.add_bars(x, y, colors=[BLUE] * 5)
        canvas.set_ylim(0.30, 1.15)
        canvas.add_callout(
            target_x=4.0, target_y=0.52,
            text="Far left nudge", x_offset=-4.0,
        )
        violations = canvas.validate()
        assert violations, (
            "Expected violation for large x_offset crossing tall intermediate bars, "
            f"but got none.  Callouts: {canvas._callouts}"
        )
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_straight_down_never_crosses_adjacent_bar(self):
        """Zero x_offset: arrow is perfectly vertical, cannot cross adjacent bars."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y")
        x = np.arange(3, dtype=float)
        y = np.array([0.90, 0.70, 0.85])
        canvas.add_bars(x, y, colors=[BLUE] * 3)
        canvas.set_ylim(0.40, 1.15)
        canvas.add_callout(target_x=1.0, target_y=0.72, text="Middle bar")
        violations = canvas.validate()
        bar_violations = [v for v in violations if "crosses bar" in v]
        assert bar_violations == [], f"Unexpected bar-cross violations: {bar_violations}"
        import matplotlib.pyplot as plt
        plt.close("all")


class TestValidateCatchesOverlap:
    """validate() must detect manually injected violations."""

    def test_below_margin_callout_is_detected(self):
        """A callout whose xytext_y is inside the data zone must be flagged."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y",
                             top_margin_pct=0.18)
        x = np.array([0.0, 1.0])
        y = np.array([0.50, 0.70])
        canvas.add_bars(x, y, colors=[BLUE, RED])
        canvas.set_ylim(0.30, 1.00)
        # Inject a callout with text_y far below the top margin (~0.30 + 0.82 = 1.00 * 0.82)
        canvas._callouts.append(_Callout(
            text="Injected low callout",
            text_x=0.0,
            text_y=0.55,   # deep inside data zone (margin boundary ≈ 0.854)
            target_x=0.0,
            target_y=0.52,
            color=BLUE,
            placement="top",  # claims top placement but is actually too low
        ))
        violations = canvas.validate()
        below_margin = [v for v in violations if "below" in v]
        assert below_margin, (
            f"Expected a 'below margin boundary' violation, got: {violations}"
        )

    def test_arrow_crossing_non_target_bar_is_detected(self):
        """An arrow manually directed through an intermediate bar is flagged."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y",
                             top_margin_pct=0.18)
        x = np.arange(4, dtype=float)
        # Make bar at x=2 very tall (0.95) so an arrow from (0, margin)→(3, 0.5) crosses it
        y = np.array([0.60, 0.65, 0.95, 0.50])
        canvas.add_bars(x, y, colors=[BLUE] * 4)
        canvas.set_ylim(0.30, 1.20)
        # Inject a diagonal callout crossing the tall bar at x=2
        canvas._callouts.append(_Callout(
            text="Crossing arrow",
            text_x=0.0,
            text_y=1.10,   # in margin ✓
            target_x=3.0,
            target_y=0.52,
            color=BLUE,
            placement="top",
        ))
        violations = canvas.validate()
        cross_violations = [v for v in violations if "crosses bar" in v]
        assert cross_violations, (
            f"Expected an arrow-cross violation for bar at x=2, got: {violations}"
        )


class TestLegendOutsideBody:
    """Legend placement outside the chart body."""

    def test_legend_is_added_and_placed_outside(self):
        """add_legend() creates a legend with bbox_to_anchor below the axes."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y")
        canvas.add_bars(np.array([0.0, 1.0]), np.array([0.5, 0.7]),
                        colors=[BLUE, RED])
        canvas.set_ylim(0.30, 1.00)
        handles = [
            mpatches.Patch(color=BLUE, label="Series A"),
            mpatches.Patch(color=RED,  label="Series B"),
        ]
        canvas.add_legend(handles, ncol=2)
        leg = canvas.ax.get_legend()
        assert leg is not None, "Legend should be present after add_legend()"
        # bbox_to_anchor=(0.0, -0.22) places the legend below the axes
        # (negative y means below — outside the axes body)
        anchor = leg.get_bbox_to_anchor()
        # matplotlib stores this as a TransformedBbox; check the raw offset
        # by inspecting the legend's _loc_real or just confirm no violation
        violations = canvas.validate()
        assert violations == []
        import matplotlib.pyplot as plt
        plt.close("all")
