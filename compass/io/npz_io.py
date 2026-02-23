"""Save and load processed activity data as compressed .npz files."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActivityData:
    """Container for processed COMPASS activity data.

    Attributes
    ----------
    activity : np.ndarray
        Activity matrix of shape ``(n_animals, n_timepoints)``.
    times : list[str]
        Time strings (``HH:MM:SS``) for each timepoint.
    dates : list[str]
        Date strings (``MM/DD/YY``) for each timepoint.
    light_on_indexes : list[int]
        Indices where lights turn on.
    light_off_indexes : list[int]
        Indices where lights turn off.
    animals : list
        Animal identifiers.
    """

    activity: np.ndarray
    times: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    light_on_indexes: list[int] = field(default_factory=list)
    light_off_indexes: list[int] = field(default_factory=list)
    animals: list = field(default_factory=list)

    @property
    def n_animals(self) -> int:
        return self.activity.shape[0]

    @property
    def n_timepoints(self) -> int:
        return self.activity.shape[1]


def save_activity_data(filepath: str | Path, data: ActivityData) -> None:
    """Save an ``ActivityData`` instance as a compressed .npz file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    data : ActivityData
        Data container to save.
    """
    np.savez_compressed(
        Path(filepath),
        activity=data.activity,
        times=data.times,
        dates=data.dates,
        light_on_indexes=data.light_on_indexes,
        light_off_indexes=data.light_off_indexes,
        animals=data.animals,
    )


def load_activity_data(filepath: str | Path) -> ActivityData:
    """Load an ``ActivityData`` instance from a .npz file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .npz file.

    Returns
    -------
    ActivityData
    """
    raw = np.load(Path(filepath), allow_pickle=True)
    return ActivityData(
        activity=raw["activity"],
        times=raw["times"].tolist() if isinstance(raw["times"], np.ndarray) else raw["times"],
        dates=raw["dates"].tolist() if isinstance(raw["dates"], np.ndarray) else raw["dates"],
        light_on_indexes=raw["light_on_indexes"].tolist(),
        light_off_indexes=raw["light_off_indexes"].tolist(),
        animals=raw["animals"].tolist() if isinstance(raw["animals"], np.ndarray) else raw["animals"],
    )
