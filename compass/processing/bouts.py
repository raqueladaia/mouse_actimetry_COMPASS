"""Bout detection: identify contiguous runs of a given activity level."""

from __future__ import annotations

import numpy as np


def collect_bout_durations(
    data: np.ndarray,
    bout_val: int,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[list[int], list[int]]]:
    """Identify bouts of a specific activity level and return their durations.

    A "bout" is a contiguous run of ``bout_val`` in the 1-D signal ``data``.
    Durations are in bins (multiply by sampling interval for real time).

    Parameters
    ----------
    data : np.ndarray
        1-D array of binned activity values.
    bout_val : int
        The activity level to detect bouts of.
    return_indices : bool
        If ``True``, also return the start/end index pairs.

    Returns
    -------
    durations : np.ndarray
        Duration of each bout in bins.
    indices : tuple of (list[int], list[int]), optional
        ``(bout_starts, bout_ends)`` — only if ``return_indices=True``.

    Raises
    ------
    ValueError
        If no bouts are found.
    """
    bstart: list[int] = []
    bend: list[int] = []

    for i in range(len(data) - 1):
        if data[i] == bout_val and data[i + 1] != bout_val:
            bend.append(i + 1)
        elif data[i] != bout_val and data[i + 1] == bout_val:
            bstart.append(i + 1)
        elif data[i] == bout_val and data[i + 1] == bout_val and i == 0:
            bstart.append(i)
        elif data[i] == bout_val and data[i + 1] == bout_val and i == len(data) - 2:
            bend.append(i + 1)

    if not bstart or not bend:
        empty = np.array([], dtype=int)
        if return_indices:
            return empty, ([], [])
        return empty

    # Trim incomplete bouts at boundaries
    if bend[0] < bstart[0]:
        bend = bend[1:]
    if bstart and bend and bend[-1] < bstart[-1]:
        bstart = bstart[:-1]

    durations = np.array(bend) - np.array(bstart)

    if return_indices:
        return durations, (bstart, bend)
    return durations


def compute_bout_metrics(
    data: np.ndarray,
    bout_val: int,
    total_length: int | None = None,
) -> dict[str, float]:
    """Compute summary bout metrics for one animal / one phase.

    Parameters
    ----------
    data : np.ndarray
        1-D binned activity array.
    bout_val : int
        Activity level to analyze.
    total_length : int, optional
        Total length for percent-time calculation. Defaults to ``len(data)``.

    Returns
    -------
    dict
        Keys: ``mean_duration``, ``n_bouts``, ``auc``, ``percent_time``.
    """
    if total_length is None:
        total_length = len(data)

    durations = collect_bout_durations(data, bout_val)

    if len(durations) == 0:
        return {
            "mean_duration": 0.0,
            "n_bouts": 0,
            "auc": 0.0,
            "percent_time": 0.0,
        }

    return {
        "mean_duration": float(np.mean(durations)),
        "n_bouts": len(durations),
        "auc": float(np.trapezoid(durations)),
        "percent_time": float(np.sum(durations) / total_length * 100),
    }
