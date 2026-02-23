"""Configuration loading and validation for COMPASS analysis parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActivityBinningConfig:
    """Activity binning thresholds.

    Parameters
    ----------
    sleep_threshold : int
        Raw values strictly below this are classified as passive (level 0).
    low_max : int
        Upper bound (inclusive) for the low-activity bin.
    medium_max : int
        Upper bound (inclusive) for the medium-activity bin.
    high_max : int
        Upper bound (inclusive) for the high-activity bin.
    """

    sleep_threshold: int = 1
    low_max: int = 20
    medium_max: int = 50
    high_max: int = 100


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration parameters applied during CSV loading.

    Parameters
    ----------
    activity_cap : int
        Values above this are clipped.  A warning is emitted when clipping
        occurs.
    noise_threshold : int
        Values strictly below this are zeroed (residual-noise silencing).
        Set to 0 to disable noise silencing.
    """

    activity_cap: int = 100
    noise_threshold: int = 0


@dataclass(frozen=True)
class CompassConfig:
    """Top-level configuration wrapper.

    Parameters
    ----------
    binning : ActivityBinningConfig
        Activity-binning thresholds.
    calibration : CalibrationConfig
        CSV calibration parameters.
    """

    binning: ActivityBinningConfig = ActivityBinningConfig()
    calibration: CalibrationConfig = CalibrationConfig()


DEFAULT_BINNING_CONFIG = ActivityBinningConfig()
DEFAULT_CALIBRATION_CONFIG = CalibrationConfig()
DEFAULT_CONFIG = CompassConfig()


def _validate_config(cfg: ActivityBinningConfig) -> None:
    """Raise ``ValueError`` if the config has inconsistent boundaries.

    Checks
    ------
    * ``sleep_threshold`` >= 0
    * boundaries in strictly ascending order: sleep_threshold <= low_max < medium_max < high_max
    """
    if cfg.sleep_threshold < 0:
        raise ValueError(
            f"sleep_threshold must be >= 0, got {cfg.sleep_threshold}"
        )
    if not (cfg.sleep_threshold <= cfg.low_max < cfg.medium_max < cfg.high_max):
        raise ValueError(
            f"Boundaries must be in ascending order: "
            f"sleep_threshold ({cfg.sleep_threshold}) <= "
            f"low_max ({cfg.low_max}) < "
            f"medium_max ({cfg.medium_max}) < "
            f"high_max ({cfg.high_max})"
        )


def _validate_calibration(cfg: CalibrationConfig) -> None:
    """Raise ``ValueError`` if calibration parameters are invalid.

    Checks
    ------
    * ``activity_cap`` > 0
    * ``noise_threshold`` >= 0
    """
    if cfg.activity_cap <= 0:
        raise ValueError(
            f"activity_cap must be > 0, got {cfg.activity_cap}"
        )
    if cfg.noise_threshold < 0:
        raise ValueError(
            f"noise_threshold must be >= 0, got {cfg.noise_threshold}"
        )


def load_config(filepath: str | Path) -> CompassConfig:
    """Load a ``CompassConfig`` from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to a JSON config file.  Expected structure::

            {
              "activity_binning": {
                "sleep_threshold": 1,
                "boundaries": {
                  "low_max": 20,
                  "medium_max": 50,
                  "high_max": 100
                }
              },
              "calibration": {
                "activity_cap": 100,
                "noise_threshold": 0
              }
            }

        The ``"calibration"`` key is optional; defaults are used when absent.

    Returns
    -------
    CompassConfig

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    KeyError
        If required keys are missing.
    ValueError
        If boundaries are not in ascending order or calibration values are
        invalid.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, encoding="utf-8") as fh:
        data = json.load(fh)

    binning = data["activity_binning"]
    boundaries = binning["boundaries"]

    binning_cfg = ActivityBinningConfig(
        sleep_threshold=int(binning["sleep_threshold"]),
        low_max=int(boundaries["low_max"]),
        medium_max=int(boundaries["medium_max"]),
        high_max=int(boundaries["high_max"]),
    )
    _validate_config(binning_cfg)

    # Calibration section is optional — use defaults if absent
    cal_data = data.get("calibration", {})
    cal_cfg = CalibrationConfig(
        activity_cap=int(cal_data.get("activity_cap", 100)),
        noise_threshold=int(cal_data.get("noise_threshold", 0)),
    )
    _validate_calibration(cal_cfg)

    return CompassConfig(binning=binning_cfg, calibration=cal_cfg)
