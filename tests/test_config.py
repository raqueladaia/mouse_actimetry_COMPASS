"""Tests for activity binning and calibration configuration loading."""

import json
import pytest

from compass.io.config import (
    ActivityBinningConfig,
    CalibrationConfig,
    CompassConfig,
    DEFAULT_BINNING_CONFIG,
    DEFAULT_CALIBRATION_CONFIG,
    DEFAULT_CONFIG,
    load_config,
    _validate_config,
    _validate_calibration,
)


class TestActivityBinningConfig:
    """Tests for ActivityBinningConfig dataclass."""

    def test_default_values(self):
        cfg = ActivityBinningConfig()
        assert cfg.sleep_threshold == 1
        assert cfg.low_max == 20
        assert cfg.medium_max == 50
        assert cfg.high_max == 100

    def test_module_default_matches(self):
        assert DEFAULT_BINNING_CONFIG == ActivityBinningConfig()

    def test_frozen(self):
        cfg = ActivityBinningConfig()
        with pytest.raises(AttributeError):
            cfg.low_max = 30  # type: ignore[misc]


class TestCalibrationConfig:
    """Tests for CalibrationConfig dataclass."""

    def test_default_values(self):
        cfg = CalibrationConfig()
        assert cfg.activity_cap == 100
        assert cfg.noise_threshold == 0

    def test_module_default_matches(self):
        assert DEFAULT_CALIBRATION_CONFIG == CalibrationConfig()

    def test_frozen(self):
        cfg = CalibrationConfig()
        with pytest.raises(AttributeError):
            cfg.activity_cap = 200  # type: ignore[misc]

    def test_custom_values(self):
        cfg = CalibrationConfig(activity_cap=150, noise_threshold=10)
        assert cfg.activity_cap == 150
        assert cfg.noise_threshold == 10


class TestCompassConfig:
    """Tests for the top-level CompassConfig wrapper."""

    def test_default_values(self):
        cfg = CompassConfig()
        assert cfg.binning == ActivityBinningConfig()
        assert cfg.calibration == CalibrationConfig()

    def test_module_default_matches(self):
        assert DEFAULT_CONFIG == CompassConfig()


class TestValidateConfig:
    """Tests for _validate_config."""

    def test_valid_config(self):
        _validate_config(ActivityBinningConfig(1, 20, 50, 100))

    def test_negative_sleep_threshold_raises(self):
        with pytest.raises(ValueError, match="sleep_threshold must be >= 0"):
            _validate_config(ActivityBinningConfig(sleep_threshold=-1))

    def test_boundaries_not_ascending_raises(self):
        with pytest.raises(ValueError, match="ascending order"):
            _validate_config(ActivityBinningConfig(low_max=50, medium_max=20, high_max=100))

    def test_equal_boundaries_raises(self):
        with pytest.raises(ValueError, match="ascending order"):
            _validate_config(ActivityBinningConfig(low_max=50, medium_max=50, high_max=100))

    def test_sleep_threshold_above_low_max_raises(self):
        with pytest.raises(ValueError, match="ascending order"):
            _validate_config(ActivityBinningConfig(sleep_threshold=25, low_max=20))


class TestValidateCalibration:
    """Tests for _validate_calibration."""

    def test_valid_calibration(self):
        _validate_calibration(CalibrationConfig(activity_cap=100, noise_threshold=0))

    def test_zero_cap_raises(self):
        with pytest.raises(ValueError, match="activity_cap must be > 0"):
            _validate_calibration(CalibrationConfig(activity_cap=0))

    def test_negative_cap_raises(self):
        with pytest.raises(ValueError, match="activity_cap must be > 0"):
            _validate_calibration(CalibrationConfig(activity_cap=-10))

    def test_negative_noise_threshold_raises(self):
        with pytest.raises(ValueError, match="noise_threshold must be >= 0"):
            _validate_calibration(CalibrationConfig(noise_threshold=-1))

    def test_noise_threshold_zero_ok(self):
        _validate_calibration(CalibrationConfig(noise_threshold=0))


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_valid_file(self, tmp_path):
        cfg_data = {
            "activity_binning": {
                "sleep_threshold": 2,
                "boundaries": {"low_max": 25, "medium_max": 60, "high_max": 100},
            }
        }
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(cfg_data))

        cfg = load_config(fp)
        assert cfg.binning.sleep_threshold == 2
        assert cfg.binning.low_max == 25
        assert cfg.binning.medium_max == 60
        assert cfg.binning.high_max == 100
        # Calibration should use defaults when key is missing
        assert cfg.calibration == DEFAULT_CALIBRATION_CONFIG

    def test_load_with_calibration(self, tmp_path):
        cfg_data = {
            "activity_binning": {
                "sleep_threshold": 1,
                "boundaries": {"low_max": 20, "medium_max": 50, "high_max": 100},
            },
            "calibration": {
                "activity_cap": 150,
                "noise_threshold": 15,
            },
        }
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(cfg_data))

        cfg = load_config(fp)
        assert cfg.calibration.activity_cap == 150
        assert cfg.calibration.noise_threshold == 15

    def test_load_calibration_missing_key_uses_defaults(self, tmp_path):
        """When calibration section is absent, defaults are used."""
        cfg_data = {
            "activity_binning": {
                "sleep_threshold": 1,
                "boundaries": {"low_max": 20, "medium_max": 50, "high_max": 100},
            }
        }
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(cfg_data))

        cfg = load_config(fp)
        assert cfg.calibration.activity_cap == 100
        assert cfg.calibration.noise_threshold == 0

    def test_load_calibration_partial_uses_defaults_for_missing(self, tmp_path):
        """When calibration has only some keys, missing ones get defaults."""
        cfg_data = {
            "activity_binning": {
                "sleep_threshold": 1,
                "boundaries": {"low_max": 20, "medium_max": 50, "high_max": 100},
            },
            "calibration": {
                "activity_cap": 200,
            },
        }
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(cfg_data))

        cfg = load_config(fp)
        assert cfg.calibration.activity_cap == 200
        assert cfg.calibration.noise_threshold == 0  # default

    def test_load_default_config_file(self):
        """Ensure the shipped default_config.json loads without error."""
        from pathlib import Path

        default_path = Path(__file__).resolve().parents[1] / "config_files" / "default_config.json"
        if default_path.exists():
            cfg = load_config(default_path)
            assert cfg.binning == DEFAULT_BINNING_CONFIG
            assert cfg.calibration == DEFAULT_CALIBRATION_CONFIG

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.json")

    def test_missing_key_raises(self, tmp_path):
        fp = tmp_path / "bad.json"
        fp.write_text(json.dumps({"activity_binning": {"sleep_threshold": 1}}))
        with pytest.raises(KeyError):
            load_config(fp)

    def test_invalid_boundaries_raises(self, tmp_path):
        cfg_data = {
            "activity_binning": {
                "sleep_threshold": 1,
                "boundaries": {"low_max": 80, "medium_max": 50, "high_max": 100},
            }
        }
        fp = tmp_path / "bad_order.json"
        fp.write_text(json.dumps(cfg_data))
        with pytest.raises(ValueError, match="ascending order"):
            load_config(fp)

    def test_invalid_calibration_raises(self, tmp_path):
        cfg_data = {
            "activity_binning": {
                "sleep_threshold": 1,
                "boundaries": {"low_max": 20, "medium_max": 50, "high_max": 100},
            },
            "calibration": {
                "activity_cap": 0,
                "noise_threshold": 0,
            },
        }
        fp = tmp_path / "bad_cal.json"
        fp.write_text(json.dumps(cfg_data))
        with pytest.raises(ValueError, match="activity_cap must be > 0"):
            load_config(fp)
