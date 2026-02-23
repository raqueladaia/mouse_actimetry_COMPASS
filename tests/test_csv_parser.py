"""Tests for load_compass_csv calibration parameters."""

import pandas as pd
import pytest

from compass.io.csv_parser import load_compass_csv


@pytest.fixture()
def simple_csv(tmp_path):
    """Create a minimal COMPASS CSV for testing calibration."""
    fp = tmp_path / "240101_1200_week1_grA.csv"
    # 5 rows, 2 PIR columns (PIR1 becomes "header", PIR2 becomes "PIR1")
    fp.write_text(
        "Time,header,PIR1,PIR2\n"
        "2024-01-01T12:00:00Z,dev,5,120\n"
        "2024-01-01T12:00:10Z,dev,30,80\n"
        "2024-01-01T12:00:20Z,dev,0,10\n"
        "2024-01-01T12:00:30Z,dev,12,50\n"
        "2024-01-01T12:00:40Z,dev,200,0\n"
    )
    return fp


class TestLoadCompassCSVCalibration:
    """Tests that load_compass_csv respects calibration parameters."""

    def test_returns_three_elements(self, simple_csv):
        result = load_compass_csv(simple_csv)
        assert len(result) == 3
        df, ts, warnings = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(ts, list)
        assert isinstance(warnings, list)

    def test_default_cap_clips_above_100(self, simple_csv):
        df, _, warnings = load_compass_csv(simple_csv)
        # PIR2 had 120 and row 5 PIR1 had 200 → both should be clipped
        assert df.max().max() <= 100
        assert len(warnings) == 1
        assert "exceeded activity_cap" in warnings[0]

    def test_custom_cap(self, simple_csv):
        df, _, warnings = load_compass_csv(simple_csv, activity_cap=150)
        # PIR2=120 is now below cap; PIR1=200 still above 150
        # Only the 200 value should trigger clipping
        assert df.max().max() <= 150
        assert len(warnings) == 1
        assert "150" in warnings[0]

    def test_no_clipping_warning_when_all_below_cap(self, simple_csv):
        df, _, warnings = load_compass_csv(simple_csv, activity_cap=300)
        assert len(warnings) == 0
        # No values clipped; max raw value was 200
        assert df.max().max() == 200

    def test_default_noise_threshold_zero_no_silencing(self, simple_csv):
        """With default noise_threshold=0, small values are preserved."""
        df, _, _ = load_compass_csv(simple_csv, activity_cap=300)
        # Raw PIR1 row 1 = 5 → should remain 5 (no silencing)
        assert 5 in df.values

    def test_noise_threshold_silences_small_values(self, simple_csv):
        df, _, _ = load_compass_csv(simple_csv, activity_cap=300, noise_threshold=15)
        # Raw PIR1 values: 5, 30, 0, 12, 200
        # After noise_threshold=15: 5→0, 30→30, 0→0, 12→0, 200→200
        header_col = df["header"]  # was PIR1
        assert header_col.iloc[0] == 0   # 5 < 15 → 0
        assert header_col.iloc[1] == 30  # 30 >= 15 → 30
        assert header_col.iloc[2] == 0   # 0 < 15 → 0
        assert header_col.iloc[3] == 0   # 12 < 15 → 0

    def test_no_baseline_subtraction(self, simple_csv):
        """Verify the old -10 baseline subtraction is gone."""
        df, _, _ = load_compass_csv(simple_csv, activity_cap=300)
        # PIR1 row 2 = 30 → should be exactly 30 (not 20)
        assert df["header"].iloc[1] == 30

    def test_floor_at_zero(self, simple_csv):
        """All values should be >= 0 after calibration."""
        df, _, _ = load_compass_csv(simple_csv)
        assert (df.values >= 0).all()
