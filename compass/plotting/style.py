"""Plot styling presets for publication-quality figures.

All figures use settings compatible with Adobe Illustrator editing
(TrueType fonts embedded in PDF).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Colorblind-friendly default palette for dynamically detected groups.
# Groups are assigned colors in the order they appear in the metadata.
DEFAULT_GROUP_PALETTE: list[str] = [
    "#999999",   # gray
    "#E69F00",   # orange
    "#009E73",   # teal
    "#CC79A7",   # purple/pink
    "#56B4E9",   # sky blue
    "#D55E00",   # vermilion
    "#F0E442",   # yellow
    "#0072B2",   # blue
]

# Default color palette (legacy, kept for backwards compat)
COLORS_KIR2_1 = {
    "control": "gray",
    "experimental": "orange",
}


def set_publication_style() -> None:
    """Configure matplotlib for publication-quality PDF output.

    Sets TrueType font embedding (type 42) so text is editable in
    Adobe Illustrator, and uses Arial 14pt as the default font.
    """
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 14,
        }
    )


def get_colors(experiment: str = "kir2_1") -> dict[str, str]:
    """Return the default color mapping.

    Parameters
    ----------
    experiment : str
        Experiment identifier. Currently only ``"kir2_1"`` is supported.

    Returns
    -------
    dict
        ``{group_name: color_string, ...}``
    """
    if experiment == "kir2_1":
        return dict(COLORS_KIR2_1)
    raise ValueError(f"Unknown experiment: {experiment!r}")
