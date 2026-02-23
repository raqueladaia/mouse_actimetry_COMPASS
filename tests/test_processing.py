"""Tests for bout detection and activity processing."""

import numpy as np
import pytest

from compass.processing.bouts import collect_bout_durations, compute_bout_metrics
from compass.processing.activity import bin_activity, remove_isolated_zeros


class TestCollectBoutDurations:
    """Tests for collect_bout_durations."""

    def test_basic_bouts(self):
        data = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        durations = collect_bout_durations(data, bout_val=0)
        assert len(durations) == 1
        np.testing.assert_array_equal(durations, [3])

    def test_single_bout(self):
        data = np.array([1, 0, 0, 0, 0, 1])
        durations = collect_bout_durations(data, bout_val=0)
        assert len(durations) == 1
        assert durations[0] == 4

    def test_multiple_bouts(self):
        data = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0])
        durations = collect_bout_durations(data, bout_val=1)
        # Bouts of 1: single 1 at index 2, double 1 at index 6-7
        assert len(durations) >= 1

    def test_active_bouts(self):
        data = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        durations = collect_bout_durations(data, bout_val=1)
        assert len(durations) == 2
        np.testing.assert_array_equal(durations, [3, 1])

    def test_return_indices(self):
        data = np.array([0, 1, 1, 0, 0, 1, 0])
        durations, (starts, ends) = collect_bout_durations(
            data, bout_val=1, return_indices=True
        )
        assert len(starts) == len(ends) == len(durations)

    def test_no_bouts_returns_empty(self):
        data = np.array([1, 1, 1, 1])
        durations = collect_bout_durations(data, bout_val=0)
        assert len(durations) == 0


class TestBinActivity:
    """Tests for bin_activity."""

    def test_binary_mode(self):
        raw = np.array([0, 5, 50, 100, 0])
        binned = bin_activity(raw, mode="binary")
        np.testing.assert_array_equal(binned, [0, 1, 1, 1, 0])

    def test_levels_mode(self):
        raw = np.array([0, 1, 20, 21, 50, 51, 100])
        binned = bin_activity(raw, mode="levels")
        np.testing.assert_array_equal(binned, [0, 1, 1, 2, 2, 3, 3])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            bin_activity(np.array([0, 1]), mode="invalid")


    def test_custom_boundaries(self):
        raw = np.array([0, 1, 10, 11, 30, 31, 60])
        binned = bin_activity(raw, mode="levels", low_max=10, medium_max=30, high_max=60)
        np.testing.assert_array_equal(binned, [0, 1, 1, 2, 2, 3, 3])

    def test_custom_sleep_threshold(self):
        raw = np.array([0, 5, 10, 15, 20])
        binned = bin_activity(raw, mode="levels", sleep_threshold=10)
        # 0, 5 are below threshold -> passive (0)
        # 10, 15, 20 are >= threshold and <= 20 -> low (1)
        np.testing.assert_array_equal(binned, [0, 0, 1, 1, 1])

    def test_custom_boundaries_binary(self):
        raw = np.array([0, 3, 5, 10])
        binned = bin_activity(raw, mode="binary", sleep_threshold=5)
        np.testing.assert_array_equal(binned, [0, 0, 1, 1])


class TestRemoveIsolatedZeros:
    """Tests for remove_isolated_zeros."""

    def test_single_isolated_zero(self):
        data = np.array([[5, 0, 5, 5, 0, 0, 5]])
        cleaned = remove_isolated_zeros(data, min_consecutive=1)
        # The single zero at index 1 should be replaced
        assert cleaned[0, 1] > 0
        # The pair of zeros at index 4-5 should remain
        assert cleaned[0, 4] == 0

    def test_no_modification_needed(self):
        data = np.array([[5, 5, 0, 0, 5]])
        cleaned = remove_isolated_zeros(data, min_consecutive=1)
        # Consecutive zeros should not be touched
        assert cleaned[0, 2] == 0
        assert cleaned[0, 3] == 0


class TestComputeBoutMetrics:
    """Tests for compute_bout_metrics."""

    def test_basic_metrics(self):
        data = np.array([1, 0, 0, 0, 1, 0, 0, 1])
        metrics = compute_bout_metrics(data, bout_val=0)
        assert metrics["n_bouts"] == 2
        assert metrics["mean_duration"] > 0
        assert 0 <= metrics["percent_time"] <= 100

    def test_no_bouts(self):
        data = np.array([1, 1, 1])
        metrics = compute_bout_metrics(data, bout_val=0)
        assert metrics["n_bouts"] == 0
        assert metrics["mean_duration"] == 0.0
