"""Significance bracket and text annotation utilities for group comparison plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def _p_to_stars(p: float) -> str:
    """Convert a p-value to star notation.

    Parameters
    ----------
    p : float
        P-value (FDR-corrected or raw).

    Returns
    -------
    str
        ``"***"`` if p < 0.001, ``"**"`` if p < 0.01, ``"*"`` if p < 0.05,
        ``"n.s."`` otherwise.
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_brackets(
    ax: Axes,
    sig_pairs: list[tuple[float, float, float]],
    fontsize: int = 8,
    bracket_height: float | None = None,
    tip_length: float | None = None,
) -> None:
    """Draw significance brackets for significant pairwise comparisons.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    sig_pairs : list of (x1, x2, p_fdr)
        Each tuple contains the x-positions of the two groups being compared
        and the FDR-corrected p-value.  Only pairs with p < 0.05 should be
        passed (non-significant pairs are omitted per CLAUDE.md).
    fontsize : int
        Font size for the p-value labels.
    bracket_height : float, optional
        Vertical step between stacked brackets.  If *None*, auto-computed
        as ~5 % of the current y-axis span.
    tip_length : float, optional
        Length of the vertical tips at each end of the bracket.  If *None*,
        defaults to ``bracket_height * 0.4``.
    """
    if not sig_pairs:
        return

    # Determine starting y position from current axes limits
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    if bracket_height is None:
        bracket_height = span * 0.06
    if tip_length is None:
        tip_length = bracket_height * 0.4

    # Start brackets just above the data
    y_start = y_max + span * 0.02

    # Sort by span width so narrower brackets are drawn first (closer to data)
    sig_pairs_sorted = sorted(sig_pairs, key=lambda t: abs(t[1] - t[0]))

    for i, (x1, x2, p_val) in enumerate(sig_pairs_sorted):
        y = y_start + i * bracket_height
        stars = _p_to_stars(p_val)
        label = f"p={p_val:.3f} {stars}"

        # Horizontal bar with vertical tips
        ax.plot(
            [x1, x1, x2, x2],
            [y - tip_length, y, y, y - tip_length],
            color="black",
            linewidth=0.8,
            clip_on=False,
        )
        # P-value text
        ax.text(
            (x1 + x2) / 2,
            y + tip_length * 0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )

    # Expand y-limits so brackets are visible
    new_top = y_start + len(sig_pairs_sorted) * bracket_height + bracket_height
    ax.set_ylim(y_min, new_top)


def add_sig_text(
    ax: Axes,
    lines: list[str],
    loc: str = "upper right",
    fontsize: int = 8,
) -> None:
    """Add a multi-line text annotation for p-values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    lines : list of str
        Each string is one line of annotation, e.g.
        ``"control vs Kir2.1: p=0.012 *"``.
    loc : str
        Anchor location (``"upper right"``, ``"upper left"``, etc.).
    fontsize : int
        Font size for the text.
    """
    if not lines:
        return

    text = "\n".join(lines)

    # Map location string to (x, y, ha, va)
    loc_map = {
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left": (0.03, 0.97, "left", "top"),
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left": (0.03, 0.03, "left", "bottom"),
    }
    x, y, ha, va = loc_map.get(loc, (0.97, 0.97, "right", "top"))

    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )
