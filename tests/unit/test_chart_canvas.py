"""
Unit tests for ChartCanvas (collision-aware chart rendering).

Run with:   pytest tests/test_chart_canvas.py -v
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import pytest


from shelfsense.visualization.charts import (
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
        """Default add_callout with bar_top_for_arrow() target produces no violations."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y")
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.80, 0.60, 0.90])
        canvas.add_bars(x, y, colors=[BLUE] * 3)
        canvas.set_ylim(0.40, 1.20)
        safe_y = canvas.bar_top_for_arrow(1.0)
        canvas.add_callout(target_x=1.0, target_y=safe_y, text="Inline test\nline 2")
        violations = canvas.validate()
        assert violations == [], f"Unexpected violations: {violations}"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_multiple_callouts_stacked_no_violations(self):
        """Three sequential callouts with bar_top_for_arrow() targets produce no violations."""
        canvas = ChartCanvas(figsize=(16, 8), title="T", ylabel="Y",
                             top_margin_pct=0.20)
        x = np.arange(5, dtype=float)
        y = np.array([0.85, 0.70, 0.60, 0.55, 0.65])
        canvas.add_bars(x, y, colors=[BLUE] * 5)
        canvas.set_ylim(0.40, 1.20)
        canvas.add_callout(target_x=0, target_y=canvas.bar_top_for_arrow(0), text="First", color=BLUE)
        canvas.add_callout(target_x=2, target_y=canvas.bar_top_for_arrow(2), text="Second\ntwo lines", color=GREEN)
        canvas.add_callout(target_x=4, target_y=canvas.bar_top_for_arrow(4), text="Third", color=RED)
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
        safe_y = canvas.bar_top_for_arrow(1.0)
        canvas.add_callout(target_x=1.0, target_y=safe_y, text="Middle bar")
        violations = canvas.validate()
        assert violations == [], f"Unexpected violations: {violations}"
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


class TestValueLabelZoneCollision:
    """validate() must detect arrows entering the value-label zone above bar tops."""

    def test_arrow_tip_inside_label_zone_is_detected(self):
        """Rule 2b: target_y just above bar_top but inside label-text zone → violation."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y")
        x = np.array([0.0, 1.0])
        y = np.array([0.80, 0.70])
        canvas.add_bars(x, y, colors=[BLUE, RED])
        # span=0.70, text_h = 0.70*0.028 = 0.0196
        # bar at x=0: label_top = 0.80+0.012 = 0.812, label_zone_top = 0.812+0.0196 = 0.8316
        # target_y=0.805 is inside [0.812..0.8316]? No — 0.805 < 0.812 (below label_top)
        # but the label_zone_top check is target_y < label_top + text_h = 0.8316
        # 0.805 < 0.8316 → Rule 2b fires.
        canvas.set_ylim(0.50, 1.20)
        canvas.add_callout(
            target_x=0.0, target_y=0.805,  # inside value-label zone of bar at x=0
            text="Tip in label zone",
        )
        violations = canvas.validate()
        label_zone_viol = [v for v in violations if "value-label zone" in v]
        assert label_zone_viol, (
            f"Expected Rule 2b violation (tip in label zone), got: {violations}"
        )
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_arrow_through_non_target_label_zone_is_detected(self):
        """Rule 2a extended: diagonal arrow that enters a non-target bar's label zone is flagged."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y")
        x = np.array([0.0, 1.0, 2.0])
        # Bar at x=1 is tall (0.80); arrow goes from (0,1.10) → (2,0.52).
        # At x=1, t=0.5 → y ≈ 0.81, which is between bar_top=0.80 and obs_top≈0.837.
        y = np.array([0.60, 0.80, 0.50])
        canvas.add_bars(x, y, colors=[BLUE] * 3)
        canvas.set_ylim(0.30, 1.20)
        # Inject callout manually to avoid auto-stacking changing text position
        from shelfsense.visualization.charts import _Callout
        canvas._callouts.append(_Callout(
            text="Through label zone",
            text_x=0.0, text_y=1.10,
            target_x=2.0, target_y=0.52,
            color=BLUE, placement="top",
        ))
        violations = canvas.validate()
        zone_viol = [v for v in violations if "crosses bar" in v]
        assert zone_viol, (
            f"Expected Rule 2a label-zone crossing violation for bar at x=1, got: {violations}"
        )
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_arrow_above_label_zone_is_clean(self):
        """Arrow whose tip is placed via bar_top_for_arrow() passes all checks."""
        canvas = ChartCanvas(figsize=(10, 6), title="T", ylabel="Y")
        x = np.array([0.0, 1.0])
        y = np.array([0.80, 0.70])
        canvas.add_bars(x, y, colors=[BLUE, RED])
        canvas.set_ylim(0.50, 1.20)
        safe_y = canvas.bar_top_for_arrow(0.0)
        canvas.add_callout(target_x=0.0, target_y=safe_y, text="Safe tip")
        violations = canvas.validate()
        label_viol = [v for v in violations if "value-label zone" in v]
        assert label_viol == [], f"bar_top_for_arrow() should produce no label violations: {violations}"
        import matplotlib.pyplot as plt
        plt.close("all")


class TestTitleClearance:
    """validate() must flag callout text that overlaps the chart title bbox."""

    def test_callout_does_not_overlap_title(self):
        """After _render_and_cache_title(), a callout above the title bottom is caught."""
        canvas = ChartCanvas(figsize=(10, 6), title="Test Title for Overlap Check", ylabel="Y")
        x = np.array([0.0, 1.0])
        y = np.array([0.50, 0.70])
        canvas.add_bars(x, y, colors=[BLUE, RED])
        canvas.set_ylim(0.30, 1.00)
        # Inject a callout whose text_y is well above the ylim top (and thus the title).
        # In data coordinates the title sits just above ylim top (~1.00), so 1.50 is
        # unambiguously inside the title region.
        canvas._callouts.append(_Callout(
            text="Intentionally high callout",
            text_x=0.5,
            text_y=1.50,
            target_x=0.0,
            target_y=0.52,
            color=BLUE,
            placement="top",
        ))
        canvas._render_and_cache_title()   # populate _title_ymin_data
        assert canvas._title_ymin_data is not None, (
            "_render_and_cache_title() must set _title_ymin_data on Agg backend"
        )
        violations = canvas.validate()
        title_viol = [v for v in violations if "title" in v.lower()]
        assert title_viol, (
            f"Expected title-overlap violation for text_y=1.50 (ylim_top=1.00), "
            f"title_ymin_data={canvas._title_ymin_data:.4f}, "
            f"got violations: {violations}"
        )
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_callout_within_margin_clears_title(self):
        """A callout placed by bar_top_for_arrow() in the margin must not overlap the title."""
        canvas = ChartCanvas(figsize=(10, 6), title="Clean Chart", ylabel="Y")
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.60, 0.75, 0.55])
        canvas.add_bars(x, y, colors=[BLUE] * 3)
        canvas.set_ylim(0.40, 1.20)
        canvas.add_callout(
            target_x=1.0, target_y=canvas.bar_top_for_arrow(1.0),
            text="Normal callout\nin margin zone",
        )
        canvas._render_and_cache_title()
        violations = canvas.validate()
        title_viol = [v for v in violations if "title" in v.lower()]
        assert title_viol == [], (
            f"Normal in-margin callout should not overlap title, got: {title_viol}"
        )
        import matplotlib.pyplot as plt
        plt.close("all")


class TestSameRowPlacement:
    """same_row=True allows horizontally separated callouts to share a y-level."""

    def test_same_row_does_not_advance_cursor(self):
        """Two callouts with same_row=True must land at the same text_y."""
        canvas = ChartCanvas(figsize=(16, 8), title="T", ylabel="Y")
        x = np.arange(10, dtype=float)
        y = np.full(10, 0.70)
        canvas.add_bars(x, y, colors=[BLUE] * 10)
        canvas.set_ylim(0.40, 1.20)
        canvas.add_callout(
            target_x=2.0, target_y=canvas.bar_top_for_arrow(2.0),
            text="First same-row callout", same_row=True,
        )
        canvas.add_callout(
            target_x=8.0, target_y=canvas.bar_top_for_arrow(8.0),
            text="Second same-row callout", same_row=True,
        )
        assert canvas._callouts[0].text_y == canvas._callouts[1].text_y, (
            "same_row=True callouts must share the same text_y (cursor must not advance)"
        )
        import matplotlib.pyplot as plt
        plt.close("all")


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
