"""Detect and correct light/dark cycle transitions from timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Default expected half-cycle duration in bins (~12 hours at ~10s/bin)
NORMAL_HALF_CYCLE_BINS = 4324


def detect_light_cycles(
    timestamps: list[pd.Timestamp],
    lights_off_time: str,
    lights_on_time: str,
) -> tuple[list[int], list[int]]:
    """Find indices of light-on and light-off transitions.

    Parameters
    ----------
    timestamps : list of pd.Timestamp
        Corrected timestamps for the recording.
    lights_off_time : str
        Time of lights off (e.g., ``"09:30:0"``).
    lights_on_time : str
        Time of lights on (e.g., ``"21:30:0"``).

    Returns
    -------
    light_on_indexes : list of int
        Indices where lights turn on.
    light_off_indexes : list of int
        Indices where lights turn off.
    """
    times = [t.strftime("%H:%M:%S") for t in timestamps]
    # Normalize the target times to match the format
    off_prefix = lights_off_time.rstrip("0").rstrip(":") if ":" in lights_off_time else lights_off_time
    on_prefix = lights_on_time.rstrip("0").rstrip(":") if ":" in lights_on_time else lights_on_time

    light_off_indexes = [i for i, t in enumerate(times) if t.startswith(lights_off_time)]
    light_on_indexes = [i for i, t in enumerate(times) if t.startswith(lights_on_time)]

    return light_on_indexes, light_off_indexes


def correct_cycle_durations(
    light_on_indexes: list[int],
    light_off_indexes: list[int],
    expected_duration: int = NORMAL_HALF_CYCLE_BINS,
) -> tuple[list[int], list[int]]:
    """Fix cycles whose duration deviates from the expected value.

    If one half-cycle has an anomalous length (e.g., due to a timestamp
    matching error), recalculate subsequent indices assuming a fixed
    half-cycle duration.

    Parameters
    ----------
    light_on_indexes : list of int
        Raw light-on indices.
    light_off_indexes : list of int
        Raw light-off indices.
    expected_duration : int
        Expected number of bins per half-cycle.

    Returns
    -------
    light_on_indexes : list of int
        Corrected light-on indices.
    light_off_indexes : list of int
        Corrected light-off indices.
    """
    on = list(light_on_indexes)
    off = list(light_off_indexes)

    # Check dark cycles (off → on)
    if len(on) > 0 and len(off) > 1:
        durations = np.array(on) - np.array(off[: len(on)])
        deviations = np.abs(durations - np.mean(durations))
        if np.sum(deviations) > 0:
            idx = int(np.argmax(deviations))
            for i in range(idx, len(on)):
                on[i] = off[i] + expected_duration
                if i + 1 < len(off):
                    off[i + 1] = on[i] + expected_duration

    # Check light cycles (on → off)
    if len(off) > 1 and len(on) > 0:
        durations = np.array(off[1:]) - np.array(on[: len(off) - 1])
        deviations = np.abs(durations - np.mean(durations))
        if np.sum(deviations) > 0:
            idx = int(np.argmax(deviations))
            for i in range(idx, len(off) - 1):
                off[i + 1] = on[i] + expected_duration

    return on, off


def add_boundary_indices(
    light_on_indexes: list[int],
    light_off_indexes: list[int],
    n_points: int,
    first_transition: str = "off",
) -> tuple[list[int], list[int]]:
    """Add recording start (index 0) and end (``n_points``) as boundaries.

    Parameters
    ----------
    light_on_indexes : list of int
    light_off_indexes : list of int
    n_points : int
        Total number of timepoints.
    first_transition : str
        ``"off"`` if the recording starts during the dark phase,
        ``"on"`` if it starts during the light phase.

    Returns
    -------
    light_on_indexes, light_off_indexes : tuple of list[int]
    """
    on = list(light_on_indexes)
    off = list(light_off_indexes)

    if first_transition == "off":
        off = [0] + off
        on = on + [n_points]
    else:
        on = [0] + on
        off = off + [n_points]

    return on, off


def segment_by_phase(
    activity: np.ndarray,
    light_on_indexes: list[int],
    light_off_indexes: list[int],
    expected_duration: int = NORMAL_HALF_CYCLE_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Split activity into dark-phase and light-phase epochs.

    Only complete 24-hour day-cycles are kept.  A day-cycle consists of
    one dark phase followed by one light phase; both must be at least
    ``expected_duration - 10`` bins long.  Incomplete boundary cycles
    (e.g. after trimming to an analysis window) are discarded.  Dark and
    light phases are filtered pair-wise so that ``dark[i]`` always
    corresponds temporally to ``light[i]``.

    Parameters
    ----------
    activity : np.ndarray
        Activity matrix of shape ``(n_animals, n_timepoints)`` or 1-D.
    light_on_indexes : list of int
    light_off_indexes : list of int
    expected_duration : int
        Expected number of bins per half-cycle (~12 h).

    Returns
    -------
    dark_epochs : np.ndarray
        Shape ``(n_animals, epoch_length, n_dark_cycles)``.
    light_epochs : np.ndarray
        Shape ``(n_animals, epoch_length, n_light_cycles)``.
    """
    dark = [activity[..., off:on] for off, on in zip(light_off_indexes, light_on_indexes[1:])]
    light = [activity[..., on:off] for on, off in zip(light_on_indexes, light_off_indexes[1:])]

    # Keep only complete 24h day-cycles (both phases >= ~12h)
    min_len = expected_duration - 10
    n_days = min(len(dark), len(light))
    keep = [
        i for i in range(n_days)
        if dark[i].shape[-1] >= min_len and light[i].shape[-1] >= min_len
    ]
    dark = [dark[i] for i in keep]
    light = [light[i] for i in keep]

    # Trim remaining epochs to uniform length (handles ±few bins from clock correction)
    if dark:
        min_dark = min(d.shape[-1] for d in dark)
        dark = [d[..., :min_dark] for d in dark]
    if light:
        min_light = min(l.shape[-1] for l in light)
        light = [l[..., :min_light] for l in light]

    dark_epochs = np.stack(dark, axis=-1) if dark else np.empty((*activity.shape[:-1], 0, 0))
    light_epochs = np.stack(light, axis=-1) if light else np.empty((*activity.shape[:-1], 0, 0))

    return dark_epochs, light_epochs
