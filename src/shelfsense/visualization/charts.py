"""
Collision-aware chart rendering for ShelfSense portfolio charts.

Design rules (enforced by the class, not the caller):
  - Chart body is DATA ONLY: bars, lines, axis labels, gridlines.
  - All annotation text lives in reserved margin zones (top, right, left).
  - Arrows travel from margin text to data points and must NOT cross bar bodies.
  - validate() detects violations before any file is written.
  - save() refuses to write if violations exist.

Usage (single-panel):
    canvas = ChartCanvas(figsize=(16, 8), title="...", ylabel="...")
    canvas.add_bars(x, y, colors)
    canvas.set_ylim(0.44, 1.15)
    canvas.add_callout(target_x=4, target_y=0.83, text="...", color=BLUE)
    canvas.add_legend(handles)
    canvas.save("reports/charts/my_chart.png")

Usage (multi-panel — blend_dynamics style):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))
    left  = ChartCanvas.from_axes(ax1, fig, title="Panel A", ylabel="...")
    right = ChartCanvas.from_axes(ax2, fig, title="Panel B", ylabel="...")
    # ... populate each panel ...
    left.save_fig(fig, "reports/charts/two_panel.png")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # noqa: F401 — re-exported for callers


# ── palette (importable by chart scripts) ────────────────────────────────────
BLUE       = "#2C6FAC"
ORANGE     = "#E07B39"
GREEN      = "#3A8C5C"
RED        = "#C0392B"
PURPLE     = "#7B3FA0"
TEAL       = "#1A7F7A"
TEAL_LIGHT = "#5AADA8"
GREY       = "#888888"
LGREY      = "#CCCCCC"

_FACECOLOR = {
    BLUE:   "#EEF4FB",
    RED:    "#FDECEA",
    GREEN:  "#E8F5E9",
    PURPLE: "#F0E8F8",
    ORANGE: "#FFF3E0",
    TEAL:   "#E0F4F3",
    GREY:   "#F5F5F5",
}


def _fc(color: str) -> str:
    return _FACECOLOR.get(color, "#F5F5F5")


# ── internal data structures ──────────────────────────────────────────────────
@dataclass
class _Bar:
    x: float
    y: float   # top of bar (= bar height for positive bars)
    w: float   # bar width


@dataclass
class _Callout:
    text: str
    text_x: float
    text_y: float
    target_x: float
    target_y: float
    color: str
    placement: str  # "top" | "right" | "left" | "free"


# ── geometry ──────────────────────────────────────────────────────────────────
def _segment_crosses_bar(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    bar_x: float,
    bar_h: float,
    bar_w: float,
) -> bool:
    """True if segment p1→p2 strictly intersects the bar body [x±w/2] × [0, h].

    Uses Liang-Barsky parametric clipping.  Returns False if the segment only
    touches a corner or edge (t0 == t1) or if one endpoint is the bar top
    (the intended target case).
    """
    x0, y0 = p1
    x1, y1 = p2
    dx, dy = x1 - x0, y1 - y0
    xmin, xmax = bar_x - bar_w / 2.0, bar_x + bar_w / 2.0
    ymin, ymax = 0.0, bar_h

    p_ = [-dx, dx, -dy, dy]
    q_ = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p_, q_):
        if abs(pi) < 1e-12:
            if qi < 0.0:
                return False  # parallel and strictly outside
        elif pi < 0.0:
            t0 = max(t0, qi / pi)
        else:
            t1 = min(t1, qi / pi)
        if t0 >= t1:
            return False

    # Require a non-degenerate interval that doesn't solely consist of the
    # start (t=0) or end (t=1) of the segment.
    return t0 < t1 and t1 > 1e-6 and t0 < 1.0 - 1e-6


# ── main class ────────────────────────────────────────────────────────────────
class ChartCanvas:
    """
    Single-panel chart with collision-aware annotation placement.

    Parameters
    ----------
    figsize : (w, h) in inches
    title : chart title string
    ylabel : y-axis label
    xlabel : x-axis label (optional)
    top_margin_pct : fraction of y range reserved for annotations (default 0.18)
    """

    def __init__(
        self,
        figsize: Tuple[float, float],
        title: str,
        ylabel: str,
        xlabel: Optional[str] = None,
        top_margin_pct: float = 0.18,
    ) -> None:
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")
        self.ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        self.ax.set_ylabel(ylabel, fontsize=11)
        if xlabel:
            self.ax.set_xlabel(xlabel, fontsize=11)
        self.ax.grid(axis="y", alpha=0.25, zorder=0)
        self._top_margin_pct = top_margin_pct
        self._init_state()

    # ── alternate constructor for multi-panel figures ─────────────────────────
    @classmethod
    def from_axes(
        cls,
        ax,
        fig,
        title: str,
        ylabel: str,
        title_color: str = "black",
        top_margin_pct: float = 0.18,
    ) -> "ChartCanvas":
        """Wrap an existing axes (e.g. one panel of plt.subplots(1, 2))."""
        obj = cls.__new__(cls)
        obj.fig = fig
        obj.ax = ax
        obj._top_margin_pct = top_margin_pct
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=title_color, pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        obj._init_state()
        return obj

    def _init_state(self) -> None:
        self._bars: List[_Bar] = []
        self._callouts: List[_Callout] = []
        self._phase_labels: List[Tuple[float, str]] = []  # (x_center, text)
        self._phase_seps: List[float] = []                # separator x positions
        self._ylim: Optional[Tuple[float, float]] = None
        self._callout_cursor: float = 0.0   # auto-stacking y cursor
        self._bar_width: float = 0.62

    # ── data methods ──────────────────────────────────────────────────────────

    def add_bars(
        self,
        x: np.ndarray,
        y: np.ndarray,
        colors: Union[List[str], str],
        width: float = 0.62,
        value_fmt: str = "{:.4f}",
        value_size: float = 9.5,
        value_pad: float = 0.012,
        value_labels: bool = True,
        zorder: int = 3,
        hatch: Optional[str] = None,
        alpha: float = 1.0,
        edgecolor: str = "white",
        linewidth: float = 0.6,
    ):
        """Draw bars and register each for collision checking.

        May be called multiple times for grouped bar charts.  Returns the
        matplotlib bar container.
        """
        self._bar_width = width
        kw: dict = dict(width=width, zorder=zorder,
                        edgecolor=edgecolor, linewidth=linewidth)
        if hatch:
            kw["hatch"] = hatch
        if alpha < 1.0:
            kw["alpha"] = alpha

        bars = self.ax.bar(x, y, color=colors, **kw)

        for rect, v in zip(bars, y):
            cx = rect.get_x() + rect.get_width() / 2
            if value_labels:
                self.ax.text(
                    cx, v + value_pad, value_fmt.format(v),
                    ha="center", va="bottom",
                    fontsize=value_size, fontweight="bold", color="#333333",
                )
            self._bars.append(_Bar(x=cx, y=float(v), w=float(width)))
        return bars

    def add_bar_label(
        self,
        bar_center_x: float,
        bar_top_y: float,
        text: str,
        color: str = "#333333",
        fontsize: float = 9.5,
        pad: float = 0.06,
    ) -> None:
        """Place a custom label above a specific bar (use when auto-label is off)."""
        self.ax.text(
            bar_center_x, bar_top_y + pad, text,
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=color,
        )

    def add_step_line(
        self,
        x_edges: np.ndarray,
        y_steps: np.ndarray,
        color: str = RED,
        lw: float = 2.0,
        label: str = "Best to date",
    ) -> None:
        """Horizontal step line (best-to-date trace) with terminal diamond."""
        self.ax.step(x_edges, y_steps, where="post",
                     color=color, lw=lw, ls="--", zorder=6, label=label)
        self.ax.plot(x_edges[-1], y_steps[-1], "D", color=color, ms=7, zorder=7)

    def add_hline(
        self,
        y: float,
        color: str = GREY,
        lw: float = 1.2,
        ls: str = "--",
        alpha: float = 0.7,
        label: Optional[str] = None,
        zorder: int = 2,
    ) -> None:
        """Horizontal reference line (e.g. SN28 reference, clip boundary)."""
        self.ax.axhline(y, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
        if label:
            self.ax.text(0.02, y + 0.004, label, color=color, fontsize=8,
                         fontstyle="italic",
                         transform=self.ax.get_yaxis_transform())

    def add_phase_separator(self, x: float) -> None:
        """Draw a vertical dashed separator line at x."""
        self.ax.axvline(x, color=LGREY, lw=1.4, ls="--", zorder=2)
        self._phase_seps.append(x)

    def add_phase_label(self, x_center: float, text: str) -> None:
        """Register a phase label to be placed in the top margin at x_center."""
        self._phase_labels.append((x_center, text))

    def set_ylim(self, ylo: float, yhi: float) -> None:
        """Set y-axis limits.  Must be called before add_callout."""
        self._ylim = (ylo, yhi)
        self.ax.set_ylim(ylo, yhi)
        span = yhi - ylo
        # Stacking cursor starts near the top of the margin zone
        self._callout_cursor = yhi - span * self._top_margin_pct * 0.38

    def set_xticks(
        self,
        ticks: np.ndarray,
        labels: List[str],
        fontsize: float = 9.5,
    ) -> None:
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(labels, fontsize=fontsize)

    # ── annotations ───────────────────────────────────────────────────────────

    def add_callout(
        self,
        target_x: float,
        target_y: float,
        text: str,
        placement: str = "top",
        color: str = BLUE,
        fontsize: float = 9,
        fontweight: str = "normal",
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        facecolor: Optional[str] = None,
        arrowlw: float = 1.3,
        arrowstyle: str = "->",
        connectionstyle: Optional[str] = None,
    ) -> None:
        """
        Add a callout annotation with a validated arrow.

        placement options
        -----------------
        "top"   Text auto-stacked in the reserved top margin.
                x = target_x + x_offset (default: straight above target).
        "right" Text at (x_data_max + 0.6 + x_offset, target_y + y_offset).
                Arrow goes left to target.
        "left"  Text at (x_data_min - 0.6 - x_offset, target_y + y_offset).
                Arrow goes right to target.
        "free"  Caller specifies text position explicitly via x_offset / y_offset
                relative to the top-margin base position.  Skips margin check in
                validate() so use only when deliberately outside the default zone.
        """
        if self._ylim is None:
            raise RuntimeError("Call set_ylim() before add_callout().")
        ylo, yhi = self._ylim
        span = yhi - ylo

        if placement == "top":
            text_x = target_x + x_offset
            text_y = self._callout_cursor
            # Advance cursor so next callout doesn't overlap
            n_lines = text.count("\n") + 1
            line_h = span * 0.032 * (fontsize / 9.0)
            self._callout_cursor += n_lines * line_h + span * 0.012

        elif placement == "right":
            xs = [b.x for b in self._bars] or [0.0]
            text_x = max(xs) + 0.6 + x_offset
            text_y = target_y + y_offset

        elif placement == "left":
            xs = [b.x for b in self._bars] or [0.0]
            text_x = min(xs) - 0.6 - x_offset
            text_y = target_y + y_offset

        elif placement == "free":
            text_x = x_offset
            text_y = self._callout_cursor + y_offset

        else:
            raise ValueError(f"Unknown placement {placement!r}. "
                             "Use 'top', 'right', 'left', or 'free'.")

        fc = facecolor or _fc(color)
        ap: dict = dict(arrowstyle=arrowstyle, color=color, lw=arrowlw)
        if connectionstyle:
            ap["connectionstyle"] = connectionstyle

        self.ax.annotate(
            text,
            xy=(target_x, target_y),
            xytext=(text_x, text_y),
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight=fontweight,
            arrowprops=ap,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=fc,
                      edgecolor=color, alpha=0.9),
            zorder=8,
        )
        self._callouts.append(_Callout(
            text=text, text_x=text_x, text_y=text_y,
            target_x=target_x, target_y=target_y,
            color=color, placement=placement,
        ))

    def add_legend(
        self,
        handles: List,
        ncol: int = 4,
        fontsize: float = 9,
        loc: str = "lower left",
        bbox_to_anchor: Tuple = (0.0, -0.22),
    ) -> None:
        """Legend always placed outside the chart body."""
        self.ax.legend(
            handles=handles,
            fontsize=fontsize,
            loc=loc,
            ncol=ncol,
            bbox_to_anchor=bbox_to_anchor,
            frameon=True,
        )

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """
        Return a list of violation strings (empty = clean chart).

        Rules checked
        -------------
        1. Every "top" callout must have xytext_y >= the top-margin boundary.
        2. No callout arrow may intersect a non-target bar body.
        """
        violations: List[str] = []
        if not self._ylim:
            return violations

        ylo, yhi = self._ylim
        margin_boundary = yhi - (yhi - ylo) * self._top_margin_pct

        for c in self._callouts:
            # Rule 1 — top-margin placement check
            if c.placement == "top" and c.text_y < margin_boundary:
                violations.append(
                    f"Callout {c.text[:40]!r}: xytext y={c.text_y:.4f} is below "
                    f"the top-margin boundary y={margin_boundary:.4f} "
                    f"(ylim top={yhi:.4f}, margin_pct={self._top_margin_pct})."
                )

            # Rule 2 — arrow-bar intersection check for all placements
            for bar in self._bars:
                # Skip the bar that the callout is pointing at
                if abs(bar.x - c.target_x) < bar.w * 0.55:
                    continue
                if _segment_crosses_bar(
                    (c.text_x, c.text_y),
                    (c.target_x, c.target_y),
                    bar.x, bar.y, bar.w,
                ):
                    violations.append(
                        f"Callout {c.text[:40]!r}: arrow from "
                        f"({c.text_x:.2f}, {c.text_y:.4f}) → "
                        f"({c.target_x:.2f}, {c.target_y:.4f}) "
                        f"crosses bar at x={bar.x:.2f} (h={bar.y:.4f})."
                    )

        return violations

    # ── finalisation ──────────────────────────────────────────────────────────

    def _render_phase_labels(self) -> None:
        """Place all registered phase labels in the top margin. Called by save()."""
        if not self._phase_labels or not self._ylim:
            return
        ylo, yhi = self._ylim
        label_y = ylo + (yhi - ylo) * (1.0 - self._top_margin_pct * 0.55)
        for x_center, text in self._phase_labels:
            self.ax.text(
                x_center, label_y, text,
                ha="center", va="bottom", fontsize=10, color="#444444",
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                          edgecolor=LGREY, alpha=0.9),
            )

    def save(
        self,
        path: str,
        dpi: int = 130,
        bottom_adjust: float = 0.22,
    ) -> None:
        """Validate then save.  Raises ValueError if any violations exist."""
        self._render_phase_labels()
        violations = self.validate()
        if violations:
            bullet = "\n".join(f"  • {v}" for v in violations)
            raise ValueError(
                f"Chart has {len(violations)} placement violation(s):\n{bullet}"
            )
        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_adjust)
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)
        print(f"Saved: {path}")

    @staticmethod
    def save_fig(
        fig,
        path: str,
        dpi: int = 130,
        bottom_adjust: float = 0.18,
    ) -> None:
        """Save a multi-panel figure (created with from_axes) to path."""
        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_adjust)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")
