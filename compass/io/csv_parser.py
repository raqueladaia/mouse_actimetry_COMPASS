"""Parse raw COMPASS actimeter CSV files.

COMPASS boxes export a single CSV per recording session.  Rows are indexed by
datetime and columns represent individual PIR sensors (one per box).

The ``Time`` column contains ISO timestamps, the ``header`` column holds a
device identifier (not activity data), and ``PIR{N}`` maps directly to box *N*.

After loading, columns are normalised so that downstream code can use the
convention: box 1 → ``header``, box N → ``PIR{N-1}``.

CSV file naming convention:
    ``{date}_{hour}_{week}_{groupids}.csv``
    e.g. ``240213_1806_week1_gr1+gr2.csv`` or ``240220_0900_week2_gr1+gr2.csv``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class CSVFileInfo:
    """Parsed metadata from a COMPASS CSV filename.

    Attributes
    ----------
    path : Path
        Full path to the CSV file.
    date : str
        Recording date string (e.g. ``"240213"``).
    hour : str
        Recording start hour (e.g. ``"1806"``).
    week : int or None
        Week number extracted from the week component (``week1``, ``w1``,
        etc.), or ``None`` for legacy filenames that lack it.
    group_ids : list[str]
        Group ID(s) parsed from the filename (split on ``+``).
    """

    path: Path
    date: str
    hour: str
    week: int | None = None
    group_ids: list[str] = field(default_factory=list)


def parse_csv_filename(filepath: str | Path) -> CSVFileInfo:
    """Parse a COMPASS CSV filename into its components.

    Supports two formats:

    * **New**: ``{date}_{hour}_{week}_{groupids}.csv``
      e.g. ``240213_1806_week1_gr1+gr2.csv``
    * **Legacy**: ``{date}_{hour}.csv``
      e.g. ``240213_1806.csv``

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file (only the filename is parsed).

    Returns
    -------
    CSVFileInfo
        Parsed filename components.  ``week`` and ``group_ids`` are
        ``None`` / empty for legacy filenames.

    Raises
    ------
    ValueError
        If the filename cannot be parsed at all (fewer than 2 underscore-
        separated parts, or the first two parts don't look like a date
        and hour).
    """
    filepath = Path(filepath)
    stem = filepath.stem
    parts = stem.split("_")

    if len(parts) < 2:
        raise ValueError(
            f"CSV filename {filepath.name!r} does not match any known format. "
            f"Expected at least {{date}}_{{hour}}.csv"
        )

    date_str = parts[0]
    hour_str = parts[1]

    # Sanity-check: date should be all digits, hour should be all digits
    if not date_str.isdigit() or not hour_str.isdigit():
        raise ValueError(
            f"CSV filename {filepath.name!r}: expected numeric date and hour, "
            f"got date={date_str!r}, hour={hour_str!r}"
        )

    # --- Legacy format: only 2 parts → {date}_{hour}.csv ---
    if len(parts) == 2:
        return CSVFileInfo(
            path=filepath,
            date=date_str,
            hour=hour_str,
            week=None,
            group_ids=[],
        )

    # --- New format: 3+ parts → try to parse week and group_ids ---
    week_str = parts[2]
    # Match "week1", "Week2", "w1", "W3", etc.
    week_match = re.match(r"(?:[wW]eek|[wW])(\d+)", week_str)

    if week_match is None:
        # Third part isn't a week identifier — treat as legacy with extra
        # underscore segments (e.g. timestamp quirks).
        return CSVFileInfo(
            path=filepath,
            date=date_str,
            hour=hour_str,
            week=None,
            group_ids=[],
        )

    week_num = int(week_match.group(1))

    # group_ids come after the week part; rejoin in case of underscores
    groupids_str = "_".join(parts[3:]) if len(parts) > 3 else ""
    group_ids = [g.strip() for g in groupids_str.split("+") if g.strip()]

    return CSVFileInfo(
        path=filepath,
        date=date_str,
        hour=hour_str,
        week=week_num,
        group_ids=group_ids,
    )


def scan_csv_folder(
    folder: str | Path,
) -> dict[int, list[CSVFileInfo]]:
    """Scan a folder for COMPASS CSV files and group them by week number.

    If filenames include week identifiers (``week1``, ``w1``, etc.) they are
    used directly.  Multiple files can share the same week number.  For legacy
    filenames without a week component, each file is assigned a sequential
    week number (1, 2, 3, …) based on sort order.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing CSV files.

    Returns
    -------
    dict[int, list[CSVFileInfo]]
        Mapping from week number to a list of ``CSVFileInfo`` objects.
        Files that cannot be parsed at all are silently skipped.
    """
    folder = Path(folder)
    result: dict[int, list[CSVFileInfo]] = {}

    csv_files = sorted(folder.glob("*.csv"))
    print(f"[scan_csv_folder] Folder: {folder}")
    print(f"[scan_csv_folder] Found {len(csv_files)} CSV file(s)")

    parsed: list[CSVFileInfo] = []
    for csv_path in csv_files:
        print(f"[scan_csv_folder]   Trying: {csv_path.name}")
        try:
            info = parse_csv_filename(csv_path)
        except ValueError as exc:
            print(f"[scan_csv_folder]     SKIP (parse failed): {exc}")
            continue
        print(f"[scan_csv_folder]     OK: week={info.week}, groups={info.group_ids}")
        parsed.append(info)

    # Assign sequential week numbers to legacy files that lack them
    has_weeks = any(info.week is not None for info in parsed)
    if not has_weeks and parsed:
        print("[scan_csv_folder]   No week identifiers in filenames — assigning sequential weeks")
        for i, info in enumerate(parsed):
            info.week = i + 1

    for info in parsed:
        if info.week is not None:
            result.setdefault(info.week, []).append(info)

    print(f"[scan_csv_folder] Result: weeks={sorted(result.keys())} ({sum(len(v) for v in result.values())} files)")
    return result


def load_compass_csv(
    filepath: str | Path,
    time_offset_min: float = 0,
    activity_cap: int = 100,
    noise_threshold: int = 0,
) -> tuple[pd.DataFrame, list[pd.Timestamp], list[str]]:
    """Load a COMPASS CSV file and apply a clock-correction offset.

    Normalises columns so that box 1 is always ``header`` and box N is
    ``PIR{N-1}``.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw COMPASS CSV file.
    time_offset_min : float
        Minutes to add to every timestamp to correct for PC clock drift.
    activity_cap : int
        Values above this are clipped.  A warning is returned when clipping
        occurs.
    noise_threshold : int
        Values strictly below this are zeroed (residual-noise silencing).
        Set to 0 to disable noise silencing.

    Returns
    -------
    table_data : pd.DataFrame
        Normalised CSV data (``header``, ``PIR1``, ``PIR2``, …).
    timestamps : list of pd.Timestamp
        Clock-corrected timestamps for each row.
    warnings : list of str
        Any warnings generated during calibration (e.g. clipping).
    """
    filepath = Path(filepath)
    warnings: list[str] = []

    # Plain integer index; timestamps in the ``Time`` column.
    # ``header`` column holds a device identifier, not activity.
    # Box N maps to ``PIR{N}`` in the raw CSV.
    table_data = pd.read_csv(filepath, header=0, engine="python",
                             delimiter=",")

    if "Time" not in table_data.columns:
        raise ValueError(
            f"CSV {filepath.name} is missing a 'Time' column."
        )
    raw_times: list[pd.Timestamp] = pd.to_datetime(
        table_data["Time"], utc=True,
    ).tolist()

    # Drop non-activity columns
    table_data = table_data.drop(columns=["Time", "header"], errors="ignore")

    # Rename PIR{N} → PIR{N-1} (and PIR1 → header) so that the
    # downstream convention  box 1 = "header", box N = "PIR{N-1}"  holds.
    rename: dict[str, str] = {"PIR1": "header"}
    for col in table_data.columns:
        m = re.match(r"^PIR(\d+)$", col)
        if m and col != "PIR1":
            rename[col] = f"PIR{int(m.group(1)) - 1}"
    table_data = table_data.rename(columns=rename)

    # Calibration: cap at activity_cap (with warning), floor at 0,
    # then optionally silence residual noise below noise_threshold.
    activity_cols = [c for c in table_data.columns if not c.startswith("LDR")]

    # Check for values exceeding the cap before clipping
    above_cap = (table_data[activity_cols] > activity_cap).sum().sum()
    if above_cap > 0:
        warnings.append(
            f"WARNING: {int(above_cap)} value(s) exceeded activity_cap "
            f"({activity_cap}) and were clipped in {filepath.name}"
        )

    table_data[activity_cols] = table_data[activity_cols].clip(upper=activity_cap)
    table_data[activity_cols] = table_data[activity_cols].clip(lower=0)

    if noise_threshold > 0:
        table_data[activity_cols] = table_data[activity_cols].where(
            table_data[activity_cols] >= noise_threshold, 0
        )

    timestamps = [t + pd.Timedelta(minutes=time_offset_min) for t in raw_times]
    return table_data, timestamps, warnings


def get_column_name(box_number: int) -> str:
    """Return the CSV column name for a given 1-based box number.

    Parameters
    ----------
    box_number : int
        Box number (1-based).

    Returns
    -------
    str
        ``"header"`` for box 1, ``"PIR{n-1}"`` for box *n*.
    """
    if box_number == 1:
        return "header"
    return f"PIR{box_number - 1}"


def extract_animal_activity(
    table_data: pd.DataFrame,
    box_numbers: Sequence[int],
) -> np.ndarray:
    """Extract activity arrays for a list of animals by box number.

    Parameters
    ----------
    table_data : pd.DataFrame
        Raw COMPASS CSV data.
    box_numbers : sequence of int
        1-based box numbers, one per animal.

    Returns
    -------
    np.ndarray
        Activity matrix of shape ``(n_animals, n_timepoints)``.
    """
    cols = [get_column_name(b) for b in box_numbers]
    return np.column_stack([table_data[c].values for c in cols]).T


def compute_sampling_stats(timestamps: list[pd.Timestamp]) -> dict:
    """Compute basic sampling interval statistics.

    Parameters
    ----------
    timestamps : list of pd.Timestamp
        Corrected timestamps.

    Returns
    -------
    dict
        Keys: ``mean_interval_s``, ``std_interval_s``, ``n_points``.
    """
    diffs = np.diff(timestamps)
    secs = np.array([d.total_seconds() for d in diffs])
    return {
        "mean_interval_s": float(np.mean(secs)),
        "std_interval_s": float(np.std(secs)),
        "n_points": len(timestamps),
    }
