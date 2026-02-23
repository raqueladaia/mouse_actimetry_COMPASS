"""Activity binning and preprocessing utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np


# Default activity level thresholds
ACTIVITY_LEVELS = {
    "passive": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

BINARY_LEVELS = {
    "passive": 0,
    "active": 1,
}


def bin_activity(
    activity: np.ndarray,
    mode: Literal["levels", "binary"] = "levels",
    sleep_threshold: int = 1,
    low_max: int = 20,
    medium_max: int = 50,
    high_max: int = 100,
) -> np.ndarray:
    """Bin raw activity values (0–100) into categorical levels.

    Parameters
    ----------
    activity : np.ndarray
        Raw activity values.
    mode : {"levels", "binary"}
        ``"levels"``: 0=passive, 1=low, 2=medium, 3=high.
        ``"binary"``: 0=passive, 1=active.
    sleep_threshold : int
        Raw values strictly below this threshold are classified as passive
        (level 0).  Default is ``1`` (only exact 0 is passive).
    low_max : int
        Upper bound (inclusive) for the low-activity bin.  Default ``20``.
    medium_max : int
        Upper bound (inclusive) for the medium-activity bin.  Default ``50``.
    high_max : int
        Upper bound (inclusive) for the high-activity bin.  Default ``100``.

    Returns
    -------
    np.ndarray
        Binned activity with same shape as input.
    """
    binned = np.zeros_like(activity)
    if mode == "binary":
        binned[activity >= sleep_threshold] = 1
    elif mode == "levels":
        binned[(activity >= sleep_threshold) & (activity <= low_max)] = 1
        binned[(activity > low_max) & (activity <= medium_max)] = 2
        binned[(activity > medium_max) & (activity <= high_max)] = 3
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'levels' or 'binary'.")
    return binned


def remove_isolated_zeros(
    data: np.ndarray,
    min_consecutive: int = 1,
) -> np.ndarray:
    """Remove isolated zero-bins that are too brief to represent true sleep.

    A single zero surrounded by nonzero values (10 s) is replaced with the
    mean of its neighbors. Optionally, pairs of consecutive zeros (20 s) can
    also be removed.

    Parameters
    ----------
    data : np.ndarray
        Activity matrix of shape ``(n_animals, n_timepoints)``.
    min_consecutive : {1, 2}
        ``1``: Remove single isolated zeros.
        ``2``: Also remove pairs of consecutive zeros.

    Returns
    -------
    np.ndarray
        Cleaned activity matrix (modified in place and returned).
    """
    if min_consecutive not in (1, 2):
        raise ValueError("min_consecutive must be 1 or 2")

    data = data.copy()
    n = data.shape[0] if data.shape[0] < data.shape[1] else data.shape[1]

    if min_consecutive >= 1:
        for i in range(n):
            a = data[i]
            for j in range(1, len(a) - 1):
                if a[j] == 0 and a[j - 1] > 0 and a[j + 1] > 0:
                    data[i, j] = int(np.ceil(np.mean([a[j - 1], a[j + 1]])))

    if min_consecutive == 2:
        for i in range(n):
            a = data[i]
            for j in range(1, len(a) - 2):
                if a[j] == 0 and a[j + 1] == 0 and a[j - 1] > 0 and a[j + 2] > 0:
                    data[i, j] = int(np.ceil(np.mean([a[j - 1], a[j + 2]])))
                    data[i, j + 1] = 1

    return data


def get_activity_levels(
    mode: Literal["levels", "binary"] = "levels",
) -> dict[str, int]:
    """Return the activity level mapping for the given mode.

    Parameters
    ----------
    mode : {"levels", "binary"}

    Returns
    -------
    dict
        Mapping of activity type names to integer values.
    """
    if mode == "levels":
        return dict(ACTIVITY_LEVELS)
    elif mode == "binary":
        return dict(BINARY_LEVELS)
    raise ValueError(f"Unknown mode: {mode!r}")
