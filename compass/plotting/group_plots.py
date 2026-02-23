"""Group comparison plots for the Kir2.1 experiment.

Plot types
----------
8.  Weekly activity profile (polar + Cartesian) — mean per group + SEM
9.  Daily sleep bout violin plots by group
10. Sleep fragmentation index — line plot across days
11. Cumulative sleep curves — accumulated sleep over 24 h per group
12. Light/dark activity ratio — line plot across days/weeks
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from compass.analysis.stats import (
    pairwise_fdr_tests,
    pairwise_bootstrap_fdr_tests,
    welch_ttest,
    fdr_correct,
)
from compass.processing.bouts import collect_bout_durations
from compass.processing.light_cycles import NORMAL_HALF_CYCLE_BINS
from compass.plotting.sig_brackets import add_sig_brackets, add_sig_text, _p_to_stars


def _truncate_groups_to_common_length(
    group_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Truncate all group arrays to the shortest length across groups.

    Different CSV files (cohorts) can produce epoch lengths that differ by
    a few bins after trimming.  This ensures all groups share the same
    time axis for plotting.
    """
    if not group_data:
        return group_data
    min_len = min(arr.shape[1] for arr in group_data.values())
    return {gr: arr[:, :min_len] for gr, arr in group_data.items()}


def _setup_polar(ax: plt.Axes) -> None:
    """Apply standard polar conventions: 0 h at top, clockwise."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


# ------------------------------------------------------------------
# 8.  Weekly activity profile (polar + Cartesian)
# ------------------------------------------------------------------

def plot_weekly_activity_profile(
    activity: np.ndarray | None = None,
    lights_off_idx: list[int] | None = None,
    animal_ids: list | None = None,
    group_map: dict[str, list] | None = None,
    colors: dict[str, str] | None = None,
    week_num: int = 0,
    *,
    group_hourly: dict[str, np.ndarray] | None = None,
) -> Figure:
    """Mean 24 h activity profile per group with SEM shading.

    Can be called in two modes:

    * **Raw data** (original): pass *activity*, *lights_off_idx*,
      *animal_ids*, *group_map*.
    * **Pre-computed**: pass *group_hourly* — a dict mapping group name
      to an ``(n_animals, 24)`` array of mean hourly activity.

    Parameters
    ----------
    activity : np.ndarray, optional
    lights_off_idx : list[int], optional
    animal_ids : list, optional
    group_map : dict[str, list], optional
    colors : dict[str, str]
    week_num : int
    group_hourly : dict[str, np.ndarray], optional
        Pre-computed per-group hourly profiles.

    Returns
    -------
    Figure
    """
    colors = colors or {}

    # --- Compute group_hourly from raw data if not provided ---
    if group_hourly is None:
        group_map = group_map or {}
        animal_ids = animal_ids or []

        day_slices: list[np.ndarray] = []
        for i in range(len(lights_off_idx) - 1):
            s, e = lights_off_idx[i], lights_off_idx[i + 1]
            if e - s > 0:
                day_slices.append(activity[:, s:e])

        if not day_slices:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No complete days", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        def _hourly(arr: np.ndarray) -> np.ndarray:
            bph = max(1, arr.shape[1] // 24)
            n_h = min(24, arr.shape[1] // bph)
            return np.array([
                np.mean(arr[:, h * bph:(h + 1) * bph], axis=1) for h in range(n_h)
            ]).T

        hourly_all = np.stack([_hourly(d) for d in day_slices], axis=0)
        hourly_mean = np.mean(hourly_all, axis=0)

        group_hourly = {}
        for gr, ani_list in group_map.items():
            idxs = [animal_ids.index(a) for a in ani_list if a in animal_ids]
            if idxs:
                group_hourly[gr] = hourly_mean[idxs]

    # --- Plot from group_hourly ---
    if not group_hourly:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n_hours = next(iter(group_hourly.values())).shape[1]
    angles = np.linspace(0, 2 * np.pi, n_hours, endpoint=False)
    hours = np.arange(n_hours)

    fig = plt.figure(figsize=(12, 5))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_cart = fig.add_subplot(1, 2, 2)
    _setup_polar(ax_polar)

    for gr, gr_data in group_hourly.items():
        m = np.mean(gr_data, axis=0)
        sem = np.std(gr_data, axis=0) / np.sqrt(gr_data.shape[0])
        c = colors.get(gr, "gray")

        m_closed = np.append(m, m[0])
        ang_closed = np.append(angles, angles[0])
        ax_polar.plot(ang_closed, m_closed, color=c, label=gr, linewidth=1.5)
        ax_polar.fill_between(ang_closed, m_closed - np.append(sem, sem[0]),
                              m_closed + np.append(sem, sem[0]),
                              color=c, alpha=0.2)

        ax_cart.plot(hours, m, color=c, label=gr, linewidth=1.5,
                     marker="o", markersize=4)
        ax_cart.fill_between(hours, m - sem, m + sem, color=c, alpha=0.2)

    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax_polar.set_xticklabels([str(h) for h in range(24)], fontsize=7)

    # --- Statistics: per-hour pairwise Welch tests, FDR across all bins×pairs ---
    groups_list = list(group_hourly.keys())
    pairs = list(combinations(groups_list, 2))
    if len(pairs) > 0 and all(
        group_hourly[g].shape[0] >= 2 for g in groups_list
    ):
        all_p: list[float] = []
        pair_hour_map: list[tuple[str, str, int]] = []
        for g1, g2 in pairs:
            for h in range(n_hours):
                _, p = welch_ttest(group_hourly[g1][:, h], group_hourly[g2][:, h])
                all_p.append(p)
                pair_hour_map.append((g1, g2, h))
        if all_p:
            _, q_vals = fdr_correct(np.array(all_p))
            for idx_q, (g1, g2, h) in enumerate(pair_hour_map):
                if q_vals[idx_q] < 0.05:
                    ax_cart.axvspan(
                        h - 0.5, h + 0.5,
                        alpha=0.2, color="yellow", zorder=0,
                    )

    ax_cart.axvspan(-0.5, 11.5, alpha=0.08, color="gray")
    ax_cart.set_xlim(-0.5, 23.5)
    ax_cart.set_xlabel("Hour from lights off")
    ax_cart.set_ylabel("Mean activity (0-100)")
    ax_cart.legend(fontsize=9, loc="upper right")
    sns.despine(ax=ax_cart)

    fig.suptitle(f"Weekly activity profile — week {week_num}", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 9.  Daily sleep bout violin plots by group
# ------------------------------------------------------------------

def plot_sleep_bout_violins_group(
    dark_epochs: np.ndarray | None = None,
    light_epochs: np.ndarray | None = None,
    animal_ids: list | None = None,
    group_map: dict[str, list] | None = None,
    colors: dict[str, str] | None = None,
    day_index: int = 0,
    week_num: int = 0,
    *,
    group_bout_durs: dict[str, dict[str, list[float]]] | None = None,
) -> tuple[Figure, dict]:
    """Violin plots of sleep bout durations: dark vs light, one violin per group.

    Can be called with raw epoch data or with pre-computed
    *group_bout_durs* (``{group: {"Dark": [dur, ...], "Light": [dur, ...]}}``).

    Returns
    -------
    tuple[Figure, dict]
        The figure and a dict mapping panel labels (e.g. ``"Dark phase"``)
        to bootstrap results from ``pairwise_bootstrap_fdr_tests``.
    """
    colors = colors or {}

    if group_bout_durs is None:
        group_map = group_map or {}
        animal_ids = animal_ids or []
        groups = list(group_map.keys())

        group_bout_durs = {}
        for gr in groups:
            idxs = [animal_ids.index(a) for a in group_map[gr] if a in animal_ids]
            phase_durs: dict[str, list[float]] = {}
            for epochs, phase_name in [(dark_epochs, "Dark"), (light_epochs, "Light")]:
                n_cyc = epochs.shape[-1]
                cyc = min(day_index, n_cyc - 1)
                all_durs: list[float] = []
                for idx in idxs:
                    d = collect_bout_durations(epochs[idx, :, cyc], bout_val=0)
                    if len(d) > 0:
                        all_durs.extend(d.tolist())
                phase_durs[phase_name] = all_durs
            group_bout_durs[gr] = phase_durs

    groups = list(group_bout_durs.keys())
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    bootstrap_by_panel: dict[str, list] = {}

    for ax, phase_name in [(axes[0], "Dark"), (axes[1], "Light")]:
        data_list: list[list[float]] = []
        palette: list[str] = []
        labels: list[str] = []

        for gr in groups:
            durs = group_bout_durs[gr].get(phase_name, [])
            data_list.append(durs if durs else [0])
            palette.append(colors.get(gr, "gray"))
            labels.append(gr)

        if data_list:
            parts = ax.violinplot(data_list, positions=range(len(groups)),
                                  showmedians=True)
            for i, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(palette[i])
                pc.set_alpha(0.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{phase_name} phase", fontsize=10)
        sns.despine(ax=ax)

        # --- Statistics: Bootstrap mean difference per pair, FDR within panel ---
        gd = {
            gr: group_bout_durs[gr].get(phase_name, [])
            for gr in groups
        }
        gd_valid = {g: v for g, v in gd.items() if len(v) >= 2}
        if len(gd_valid) >= 2:
            results = pairwise_bootstrap_fdr_tests(gd_valid)
            bootstrap_by_panel[f"{phase_name} phase"] = results
            sig = [
                (groups.index(g1), groups.index(g2), p)
                for g1, g2, p, is_sig, _ in results if is_sig
            ]
            add_sig_brackets(ax, sig)

    axes[0].set_ylabel("Sleep bout duration (bins)")
    fig.suptitle(f"Sleep bouts by group — week {week_num}, day {day_index + 1}",
                 fontsize=11)
    fig.tight_layout()
    return fig, bootstrap_by_panel


# ------------------------------------------------------------------
# 10. Sleep fragmentation index
# ------------------------------------------------------------------

def plot_sleep_fragmentation(
    activity_bin: np.ndarray | None = None,
    lights_off_idx: list[int] | None = None,
    lights_on_idx: list[int] | None = None,
    animal_ids: list | None = None,
    group_map: dict[str, list] | None = None,
    colors: dict[str, str] | None = None,
    week_num: int = 0,
    *,
    group_frag: dict[str, np.ndarray] | None = None,
) -> Figure:
    """Line plot of sleep fragmentation index across days.

    Can be called with raw data or with pre-computed *group_frag*
    (``{group: ndarray(n_animals, n_days)}``).

    Returns
    -------
    Figure
    """
    colors = colors or {}

    if group_frag is None:
        group_map = group_map or {}
        animal_ids = animal_ids or []
        n_days = len(lights_off_idx) - 1
        if n_days < 1:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No complete days", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        group_frag = {}
        for gr in group_map:
            idxs = [animal_ids.index(a) for a in group_map[gr] if a in animal_ids]
            if not idxs:
                continue
            frag_per_day = np.full((len(idxs), n_days), np.nan)
            for d in range(n_days):
                s, e = lights_off_idx[d], lights_off_idx[d + 1]
                for j, idx in enumerate(idxs):
                    day_data = activity_bin[idx, s:e]
                    durs = collect_bout_durations(day_data, bout_val=0)
                    total_sleep = float(np.sum(day_data == 0))
                    if total_sleep > 0 and len(durs) > 0:
                        frag_per_day[j, d] = len(durs) / total_sleep
                    else:
                        frag_per_day[j, d] = 0.0
            group_frag[gr] = frag_per_day

    if not group_frag:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n_days = max(arr.shape[1] for arr in group_frag.values())
    fig, ax = plt.subplots(figsize=(8, 5))

    for gr, frag_arr in group_frag.items():
        n_d = frag_arr.shape[1]
        m = np.nanmean(frag_arr, axis=0)
        sem = np.nanstd(frag_arr, axis=0) / np.sqrt(frag_arr.shape[0])
        days = np.arange(1, n_d + 1)
        c = colors.get(gr, "gray")
        ax.plot(days, m, color=c, marker="o", markersize=4, label=gr, linewidth=1.5)
        ax.fill_between(days, m - sem, m + sem, color=c, alpha=0.2)

    # --- Statistics: Welch's t-test on per-animal mean across days ---
    animal_means = {
        gr: np.nanmean(frag_arr, axis=1)
        for gr, frag_arr in group_frag.items()
        if frag_arr.shape[0] >= 2
    }
    if len(animal_means) >= 2:
        results = pairwise_fdr_tests(animal_means, test="welch")
        sig_lines = [
            f"{g1} vs {g2}: p={p:.3f} {_p_to_stars(p)}"
            for g1, g2, p, is_sig in results if is_sig
        ]
        if sig_lines:
            add_sig_text(ax, sig_lines, loc="upper right")

    ax.set_xlabel("Day")
    ax.set_ylabel("Fragmentation index\n(n_bouts / sleep_time)")
    ax.legend(fontsize=9, loc="best")
    sns.despine(ax=ax)
    fig.suptitle(f"Sleep fragmentation — week {week_num}", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 11. Cumulative sleep curves
# ------------------------------------------------------------------

def plot_cumulative_sleep(
    activity_bin: np.ndarray | None = None,
    lights_off_idx: list[int] | None = None,
    animal_ids: list | None = None,
    group_map: dict[str, list] | None = None,
    colors: dict[str, str] | None = None,
    week_num: int = 0,
    *,
    group_cumul: dict[str, np.ndarray] | None = None,
) -> Figure:
    """Overlay of total accumulated sleep time across 24 h per group.

    Can be called with raw data or with pre-computed *group_cumul*
    (``{group: ndarray(n_animals, day_len)}``).

    Returns
    -------
    Figure
    """
    colors = colors or {}

    if group_cumul is None:
        group_map = group_map or {}
        animal_ids = animal_ids or []
        n_days = len(lights_off_idx) - 1
        if n_days < 1:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        day_len = min(lights_off_idx[d + 1] - lights_off_idx[d] for d in range(n_days))

        group_cumul = {}
        for gr in group_map:
            idxs = [animal_ids.index(a) for a in group_map[gr] if a in animal_ids]
            if not idxs:
                continue
            cumsum_all: list[np.ndarray] = []
            for idx in idxs:
                per_day: list[np.ndarray] = []
                for d in range(n_days):
                    s = lights_off_idx[d]
                    day_data = activity_bin[idx, s:s + day_len]
                    per_day.append(np.cumsum(day_data == 0).astype(float))
                cumsum_all.append(np.mean(per_day, axis=0))
            group_cumul[gr] = np.array(cumsum_all)

    if not group_cumul:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    group_cumul = _truncate_groups_to_common_length(group_cumul)
    day_len = next(iter(group_cumul.values())).shape[1]
    fig, ax = plt.subplots(figsize=(8, 5))

    for gr, arr in group_cumul.items():
        m = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
        x = np.arange(arr.shape[1])
        c = colors.get(gr, "gray")
        ax.plot(x, m, color=c, label=gr, linewidth=1.5)
        ax.fill_between(x, m - sem, m + sem, color=c, alpha=0.2)

    # --- Statistics: Welch's t-test at endpoint per pair ---
    endpoint_data = {
        gr: arr[:, -1]
        for gr, arr in group_cumul.items()
        if arr.shape[0] >= 2
    }
    if len(endpoint_data) >= 2:
        results = pairwise_fdr_tests(endpoint_data, test="welch")
        sig_lines = [
            f"{g1} vs {g2}: p={p:.3f} {_p_to_stars(p)}"
            for g1, g2, p, is_sig in results if is_sig
        ]
        if sig_lines:
            add_sig_text(ax, sig_lines, loc="upper left")

    half = day_len // 2
    ax.axvspan(0, half, alpha=0.08, color="gray")
    ax.set_xlabel("Bin (from lights off)")
    ax.set_ylabel("Cumulative sleep (bins)")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.suptitle(f"Cumulative sleep — week {week_num}", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 11a. Cumulative sleep over 24h (descriptive, x-axis in hours)
# ------------------------------------------------------------------

def plot_cumulative_sleep_24h(
    group_cumul: dict[str, np.ndarray],
    colors: dict[str, str],
    week_num: int = 0,
) -> Figure:
    """Full 24 h cumulative sleep with dark/light phase background shading.

    Parameters
    ----------
    group_cumul : dict[str, np.ndarray]
        ``{group: ndarray(n_animals, day_len)}`` — per-animal cumulative
        sleep already averaged across days.
    colors : dict[str, str]
    week_num : int

    Returns
    -------
    Figure
    """
    if not group_cumul:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    group_cumul = _truncate_groups_to_common_length(group_cumul)
    day_len = next(iter(group_cumul.values())).shape[1]
    x_hours = np.linspace(0, 24, day_len, endpoint=False)
    half = day_len // 2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Phase shading — dark phase (first ~12 h)
    ax.axvspan(0, x_hours[half] if half < day_len else 12,
               alpha=0.08, color="gray")

    for gr, arr in group_cumul.items():
        m = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
        c = colors.get(gr, "gray")
        ax.plot(x_hours, m, color=c, label=gr, linewidth=1.5)
        ax.fill_between(x_hours, m - sem, m + sem, color=c, alpha=0.2)

    ax.set_xlabel("Hour from lights off")
    ax.set_ylabel("Cumulative sleep (bins)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.suptitle(f"Cumulative sleep over 24h — week {week_num}", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 11b. Cumulative sleep by phase (dark + light panels)
# ------------------------------------------------------------------

def plot_cumulative_sleep_phases(
    group_cumul_dark: dict[str, np.ndarray],
    group_cumul_light: dict[str, np.ndarray],
    colors: dict[str, str],
    week_num: int = 0,
) -> Figure:
    """Two-panel cumulative sleep: dark phase (left) and light phase (right).

    Parameters
    ----------
    group_cumul_dark : dict[str, np.ndarray]
        ``{group: ndarray(n_animals, half_cycle_len)}`` — per-animal
        cumulative sleep within the dark phase, averaged across days.
    group_cumul_light : dict[str, np.ndarray]
        Same for the light phase.
    colors : dict[str, str]
    week_num : int

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ylims = {"Dark phase": (0, 10000), "Light phase": (0, 5000)}

    for ax, phase_data, phase_name in [
        (axes[0], group_cumul_dark, "Dark phase"),
        (axes[1], group_cumul_light, "Light phase"),
    ]:
        if not phase_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(phase_name, fontsize=10)
            sns.despine(ax=ax)
            continue

        phase_data = _truncate_groups_to_common_length(phase_data)
        phase_len = next(iter(phase_data.values())).shape[1]
        x_hours = np.linspace(0, 12, phase_len, endpoint=False)

        for gr, arr in phase_data.items():
            m = np.mean(arr, axis=0)
            sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
            c = colors.get(gr, "gray")
            ax.plot(x_hours, m, color=c, label=gr, linewidth=1.5)
            ax.fill_between(x_hours, m - sem, m + sem, color=c, alpha=0.2)

        ax.set_xlabel("Hour")
        ax.set_xlim(0, 12)
        ax.set_ylim(ylims[phase_name])
        ax.set_xticks(range(0, 13, 2))
        ax.set_title(phase_name, fontsize=10)
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Cumulative sleep (bins)")
    fig.suptitle(f"Cumulative sleep by phase — week {week_num}", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 12. Light / dark activity ratio
# ------------------------------------------------------------------

def plot_light_dark_ratio(
    ratio_data: dict[str, dict[int, list[list[float]]]],
    colors: dict[str, str],
    week_nums: list[int],
) -> Figure:
    """Line plot of (dark activity) / (light activity) ratio across days/weeks.

    Parameters
    ----------
    ratio_data : dict
        ``{group_name: {week_num: [[ani1_day1, ani1_day2, ...], ...], ...}}``.
        Each inner list is one animal's daily ratios for that week.
    colors : dict[str, str]
    week_nums : list[int]

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_offset = 0
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    groups = list(ratio_data.keys())

    for wk in sorted(week_nums):
        n_days_max = 0
        for gr in groups:
            if wk in ratio_data[gr] and ratio_data[gr][wk]:
                n_days_max = max(n_days_max, max(len(a) for a in ratio_data[gr][wk]))

        for gr in groups:
            if wk not in ratio_data[gr] or not ratio_data[gr][wk]:
                continue
            # Stack per-animal arrays → (n_animals, n_days), compute mean ± SEM
            arr = np.array(ratio_data[gr][wk])  # (n_animals, n_days)
            m = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            n_d = len(m)
            days_x = np.arange(x_offset, x_offset + n_d)
            c = colors.get(gr, "gray")
            ax.plot(days_x, m, color=c, marker="o", markersize=3,
                    linewidth=1.2, label=gr if wk == sorted(week_nums)[0] else None)
            ax.fill_between(days_x, m - sem, m + sem, color=c, alpha=0.2)

        mid = x_offset + n_days_max / 2 - 0.5
        tick_positions.append(mid)
        tick_labels.append(f"wk {wk}")
        x_offset += n_days_max + 1  # gap between weeks

    # --- Statistics: Welch's t-test per pair per week ---
    sig_lines: list[str] = []
    for wk in sorted(week_nums):
        animal_means: dict[str, list[float]] = {}
        for gr in groups:
            if wk not in ratio_data[gr] or not ratio_data[gr][wk]:
                continue
            arr = np.array(ratio_data[gr][wk])
            # Per-animal mean across days within the week
            per_ani = np.nanmean(arr, axis=1)
            animal_means[gr] = per_ani.tolist()
        valid = {g: np.asarray(v) for g, v in animal_means.items() if len(v) >= 2}
        if len(valid) >= 2:
            results = pairwise_fdr_tests(valid, test="welch")
            for g1, g2, p, is_sig in results:
                if is_sig:
                    sig_lines.append(f"wk{wk} {g1} vs {g2}: p={p:.3f} {_p_to_stars(p)}")
    if sig_lines:
        add_sig_text(ax, sig_lines, loc="upper right")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Dark / Light activity ratio")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    sns.despine(ax=ax)
    fig.suptitle("Light/dark activity ratio", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 13. Weekly sleep bout violin plots (weeks × phases grid)
# ------------------------------------------------------------------

def plot_weekly_sleep_bout_violins(
    bout_data: dict[int, dict[str, dict[str, list[float]]]],
    colors: dict[str, str],
    week_nums: list[int],
) -> tuple[Figure, dict]:
    """Violin plots of sleep bout durations pooled across all days within each week.

    Parameters
    ----------
    bout_data : dict
        ``{week_num: {group_name: {"Dark": [dur, ...], "Light": [dur, ...]}}}``.
        Durations are pre-pooled across all days within each week.
    colors : dict[str, str]
        Group name → colour.
    week_nums : list[int]
        Ordered list of week numbers to display (one row each).

    Returns
    -------
    tuple[Figure, dict]
        The figure and a dict mapping panel labels (e.g. ``"Week 1 Dark phase"``)
        to bootstrap results from ``pairwise_bootstrap_fdr_tests``.
    """
    n_weeks = len(week_nums)
    groups = list(next(iter(bout_data.values())).keys())
    fig, axes = plt.subplots(
        n_weeks, 2,
        figsize=(10, 4 * n_weeks),
        sharey=True,
        squeeze=False,
    )
    bootstrap_by_panel: dict[str, list] = {}

    for i_wk, wk in enumerate(week_nums):
        wk_data = bout_data.get(wk, {})
        for j_ph, phase_name in enumerate(("Dark", "Light")):
            ax = axes[i_wk, j_ph]
            data_list: list[list[float]] = []
            palette: list[str] = []
            labels: list[str] = []

            for gr in groups:
                durs = wk_data.get(gr, {}).get(phase_name, [])
                data_list.append(durs if durs else [0])
                palette.append(colors.get(gr, "gray"))
                labels.append(gr)

            if data_list:
                parts = ax.violinplot(
                    data_list, positions=range(len(groups)), showmedians=True,
                )
                for k, pc in enumerate(parts.get("bodies", [])):
                    pc.set_facecolor(palette[k])
                    pc.set_alpha(0.5)

            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(labels, fontsize=9)
            sns.despine(ax=ax)

            # --- Statistics: Bootstrap mean difference per pair, FDR within panel ---
            gd = {
                gr: wk_data.get(gr, {}).get(phase_name, [])
                for gr in groups
            }
            gd_valid = {g: v for g, v in gd.items() if len(v) >= 2}
            if len(gd_valid) >= 2:
                results = pairwise_bootstrap_fdr_tests(gd_valid)
                bootstrap_by_panel[f"Week {wk} {phase_name} phase"] = results
                sig = [
                    (groups.index(g1), groups.index(g2), p)
                    for g1, g2, p, is_sig, _ in results if is_sig
                ]
                add_sig_brackets(ax, sig)

            # Column headers on top row only
            if i_wk == 0:
                ax.set_title(f"{phase_name} phase", fontsize=10)
            # Row labels on left column only
            if j_ph == 0:
                ax.set_ylabel(f"Week {wk}", fontsize=10)

    fig.suptitle("Weekly sleep bout durations (pooled across days)", fontsize=11)
    fig.tight_layout()
    return (fig, bootstrap_by_panel)


# ------------------------------------------------------------------
# 14. Bootstrap histogram companion figures
# ------------------------------------------------------------------

def plot_bootstrap_histograms(
    bootstrap_by_panel: dict[str, list[tuple[str, str, float, bool, dict]]],
    title: str = "",
) -> Figure | None:
    """Bootstrap histogram companion figure for violin plot figures.

    Parameters
    ----------
    bootstrap_by_panel : dict
        Mapping of panel label (e.g., ``"Dark phase"``, ``"Light phase"``)
        to the list of pairwise bootstrap results from
        ``pairwise_bootstrap_fdr_tests``.  Each result is
        ``(g1, g2, p_fdr, significant, result_dict)``.
    title : str
        Figure super-title.

    Returns
    -------
    Figure or None
        ``None`` if there are no results to plot.
    """
    panels = list(bootstrap_by_panel.keys())
    if not panels:
        return None

    first_results = next(iter(bootstrap_by_panel.values()))
    if not first_results:
        return None

    n_panels = len(panels)
    n_pairs = len(first_results)

    fig, axes = plt.subplots(
        n_panels, n_pairs,
        figsize=(5 * n_pairs, 4 * n_panels),
        squeeze=False,
    )

    for i_panel, panel_label in enumerate(panels):
        results = bootstrap_by_panel[panel_label]
        for i_pair, (g1, g2, p_fdr, _is_sig, res_dict) in enumerate(results):
            ax = axes[i_panel, i_pair]

            boot_diffs = res_dict.get("bootstrap_diffs", np.array([]))
            obs_diff = res_dict.get("observed_diff", 0.0)
            lower_ci = res_dict.get("lower_ci", 0.0)
            upper_ci = res_dict.get("upper_ci", 0.0)

            if len(boot_diffs) > 0:
                ax.hist(boot_diffs, bins=50, color="gray", alpha=0.5)
                ax.axvline(0, color="red", linestyle="--", linewidth=1)
                ax.axvline(obs_diff, color="black", linestyle="--", linewidth=1)
                ax.axvline(lower_ci, color="blue", linestyle="--", linewidth=1)
                ax.axvline(upper_ci, color="blue", linestyle="--", linewidth=1)

                stars = _p_to_stars(p_fdr)
                ann_text = (
                    f"Obs diff: {obs_diff:.3f}\n"
                    f"95% CI: [{lower_ci:.3f}, {upper_ci:.3f}]\n"
                    f"p(FDR)={p_fdr:.4f} {stars}"
                )
                ax.text(
                    0.97, 0.97, ann_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
                )
            else:
                ax.text(
                    0.5, 0.5, "Insufficient data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray",
                )

            ax.set_xlabel("Difference in means")
            sns.despine(ax=ax)

            # Column titles on top row
            if i_panel == 0:
                ax.set_title(f"{g1} vs {g2}", fontsize=10)
            # Row labels on left column
            if i_pair == 0:
                ax.set_ylabel(f"{panel_label}\nFrequency", fontsize=10)
            else:
                ax.set_ylabel("Frequency")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
