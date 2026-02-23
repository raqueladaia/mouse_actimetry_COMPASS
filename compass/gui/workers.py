"""QThread workers that run experiment pipelines off the main thread."""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from PyQt6.QtCore import QThread, pyqtSignal

from compass.io.config import (
    ActivityBinningConfig,
    CalibrationConfig,
    DEFAULT_BINNING_CONFIG,
    DEFAULT_CALIBRATION_CONFIG,
)
from compass.io.csv_parser import (
    load_compass_csv,
    extract_animal_activity,
    compute_sampling_stats,
    parse_csv_filename,
    scan_csv_folder,
)
from compass.io.metadata import load_metadata, get_animals_by_group
from compass.io.npz_io import ActivityData, save_activity_data
from compass.processing.activity import bin_activity, remove_isolated_zeros, get_activity_levels
from compass.processing.bouts import collect_bout_durations, compute_bout_metrics
from compass.processing.light_cycles import (
    detect_light_cycles,
    correct_cycle_durations,
    add_boundary_indices,
    segment_by_phase,
    NORMAL_HALF_CYCLE_BINS,
)
from compass.analysis.stats import welch_ttest, mann_whitney, fdr_correct, bootstrap_mean_difference, pairwise_fdr_tests
from compass.plotting.sig_brackets import add_sig_brackets
from compass.plotting.style import set_publication_style, get_colors
from compass.plotting.control_plots import (
    plot_hourly_activity_bars,
    plot_daily_activity_overlay,
    plot_sleep_bout_violins_animal,
    plot_actogram,
)
from compass.plotting.group_plots import (
    plot_weekly_activity_profile,
    plot_sleep_bout_violins_group,
    plot_sleep_fragmentation,
    plot_cumulative_sleep,
    plot_cumulative_sleep_24h,
    plot_cumulative_sleep_phases,
    plot_light_dark_ratio,
    plot_weekly_sleep_bout_violins,
    plot_bootstrap_histograms,
)


# ---------------------------------------------------------------------------
# Shared parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineParams:
    data_folder: str = ""
    metadata_file: str = ""
    output_folder: str = ""
    file_offsets: dict[str, float] = field(default_factory=dict)  # {filename: offset_min}
    lights_off_time: str = "09:30:0"
    lights_on_time: str = "21:30:0"
    selected_weeks: list[int] = field(default_factory=list)
    analysis_mode: Literal["levels", "binary"] = "levels"
    binning_config: ActivityBinningConfig = field(default_factory=lambda: DEFAULT_BINNING_CONFIG)
    calibration_config: CalibrationConfig = field(default_factory=lambda: DEFAULT_CALIBRATION_CONFIG)
    colors: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Kir2.1 worker
# ---------------------------------------------------------------------------

class Kir21Worker(QThread):
    """Run the full Kir2.1 pipeline: data collection + activity profiles + bout analysis."""

    progress = pyqtSignal(int)      # 0-100
    log = pyqtSignal(str)
    figure_ready = pyqtSignal(str, object)  # (name, Figure)
    stats_text = pyqtSignal(str)
    finished_ok = pyqtSignal()
    finished_err = pyqtSignal(str)

    def __init__(self, params: PipelineParams, parent=None) -> None:
        super().__init__(parent)
        self.p = params

    def run(self) -> None:  # noqa: C901 — pipeline logic is inherently long
        try:
            self._run_pipeline()
            self.finished_ok.emit()
        except Exception:
            self.finished_err.emit(traceback.format_exc())

    def _run_pipeline(self) -> None:
        p = self.p
        set_publication_style()
        out = Path(p.output_folder)
        out.mkdir(parents=True, exist_ok=True)

        meta = load_metadata(p.metadata_file)
        groups = meta["group_name"].unique().tolist()
        animals = meta["mouse"].unique().tolist()
        colors = p.colors if p.colors else get_colors("kir2_1")

        # Scan CSV files by week using filename parser
        data_dir = Path(p.data_folder)
        weeks_map = scan_csv_folder(data_dir)
        if not weeks_map:
            # Fallback: try plain sorted glob
            csv_files = sorted(data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {data_dir}")
            weeks_map = {1: []}  # fallback structure

        # Determine which weeks to process
        selected = p.selected_weeks if p.selected_weeks else sorted(weeks_map.keys())

        # Build a flat list of CSV files to process (only selected weeks)
        csv_files_to_process: list[Path] = []
        for wk in selected:
            if wk in weeks_map:
                for info in weeks_map[wk]:
                    csv_files_to_process.append(info.path)

        if not csv_files_to_process:
            raise FileNotFoundError(f"No CSV files found for selected weeks {selected}")

        n_files = len(csv_files_to_process)
        stats_lines: list[str] = []
        total_steps = n_files + len(selected)  # data collection + bout analysis per week
        step = 0

        sni_date_str = meta["sni_date"].unique()[0].strftime("%D")

        # ---- PHASE 1: Data Collection ----
        self.log.emit("=== Phase 1: Data Collection ===")
        # Multiple CSV files can share the same week number (different cohorts
        # recorded at different times).  Each file gets its own .npz.
        npz_files: dict[int, list[Path]] = {}
        file_counter: dict[int, int] = {}  # per-week file index for unique names

        for csv_path in csv_files_to_process:
            offset = p.file_offsets.get(csv_path.name, 0.0)
            self.log.emit(f"Loading {csv_path.name} (offset={offset} min)...")
            table_data, timestamps, csv_warnings = load_compass_csv(
                csv_path,
                time_offset_min=offset,
                activity_cap=p.calibration_config.activity_cap,
                noise_threshold=p.calibration_config.noise_threshold,
            )
            for w in csv_warnings:
                self.log.emit(f"  {w}")

            dates = [t.strftime("%D") for t in timestamps]
            times = [t.strftime("%H:%M:%S") for t in timestamps]
            n_points = len(times)

            sampling = compute_sampling_stats(timestamps)
            self.log.emit(f"  {n_points} points, mean interval {sampling['mean_interval_s']:.2f}s")

            # Parse week from filename
            try:
                file_info = parse_csv_filename(csv_path)
                week_num = file_info.week
            except ValueError:
                # Fallback to SNI-based week calculation
                days_after_sni = (pd.to_datetime(dates[0]) - pd.to_datetime(sni_date_str)).days
                week_num = days_after_sni // 7

            self.log.emit(f"  Week {week_num}")

            # Light cycle detection
            light_on_idx, light_off_idx = detect_light_cycles(timestamps, p.lights_off_time, p.lights_on_time)
            light_on_idx, light_off_idx = correct_cycle_durations(light_on_idx, light_off_idx)
            light_on_idx, light_off_idx = add_boundary_indices(light_on_idx, light_off_idx, n_points, first_transition="off")

            # Analysis window
            days_after_sni = (pd.to_datetime(dates[0]) - pd.to_datetime(sni_date_str)).days
            weeks_after_sni = days_after_sni // 7
            if weeks_after_sni < 3:
                start = light_off_idx[2]
            else:
                start = light_off_idx[-5]
            end = light_off_idx[-1]

            # Extract activity — filter to animals whose group_id matches this file
            try:
                file_info = parse_csv_filename(csv_path)
                file_group_ids = file_info.group_ids

                # Flexible group_id matching: filenames may use prefixed ids
                # (e.g. "gr1", "gr2") while metadata stores plain ints (1, 2, 3).
                # Build a combined set of candidate values to match against.
                match_candidates: set = set(file_group_ids)
                for gid in file_group_ids:
                    num = re.sub(r"[^0-9]", "", gid)
                    if num:
                        match_candidates.add(int(num))
                        match_candidates.add(num)  # string form too

                mask = meta["group_id"].isin(match_candidates) | meta["group_id"].astype(str).isin(match_candidates)
                file_animals = meta.loc[mask, "mouse"].tolist()
                box_numbers = meta.loc[meta["mouse"].isin(file_animals), "actimeter_box"].tolist()
                self.log.emit(
                    f"  Matched group_ids {file_group_ids} -> {len(file_animals)} animals, "
                    f"boxes {box_numbers}"
                )
            except (ValueError, KeyError) as exc:
                # Fallback: use all animals
                self.log.emit(f"  Group matching failed ({exc}), using all animals")
                file_animals = animals
                box_numbers = meta["actimeter_box"].tolist()

            if not box_numbers:
                raise ValueError(
                    f"No animals matched for {csv_path.name}. "
                    f"Filename group_ids={file_group_ids!r}, "
                    f"metadata group_id values={meta['group_id'].unique().tolist()!r}"
                )

            activity_all = extract_animal_activity(table_data, box_numbers)

            # Trim to analysis window
            activity_trim = activity_all[:, start:end + 1]
            times_trim = times[start:end + 1]
            dates_trim = dates[start:end + 1]
            on_trim = [l - start for l in light_on_idx if start <= l <= end]
            off_trim = [l - start for l in light_off_idx if start <= l <= end]

            # Save .npz — unique name per CSV file within the same week
            fidx = file_counter.get(week_num, 0)
            file_counter[week_num] = fidx + 1
            suffix = "" if fidx == 0 else f"_{fidx}"
            npz_path = out / f"activity_animals_week{week_num}{suffix}.npz"
            data_obj = ActivityData(
                activity=activity_trim,
                times=times_trim,
                dates=dates_trim,
                light_on_indexes=on_trim,
                light_off_indexes=off_trim,
                animals=file_animals,
            )
            save_activity_data(npz_path, data_obj)
            npz_files.setdefault(week_num, []).append(npz_path)
            self.log.emit(f"  Saved {npz_path.name}")

            # Data overview figure
            n_ani_fig = len(file_animals)
            fig, axes = plt.subplots(n_ani_fig, 1,
                                     figsize=(12, max(3, 2 * n_ani_fig)),
                                     squeeze=False, sharex=True)
            fig.suptitle(f"Activity — week {week_num} ({csv_path.stem})")
            x_range = np.arange(n_points)
            for i_ani, ani in enumerate(file_animals):
                gr = meta.loc[meta["mouse"] == ani, "group_name"].values[0]
                sex_vals = meta.loc[meta["mouse"] == ani, "sex"]
                sex = sex_vals.values[0] if len(sex_vals) > 0 else "?"
                ax = axes[i_ani, 0]
                ax.plot(x_range, activity_all[i_ani], color=colors.get(gr, "gray"), alpha=0.3, linewidth=0.5)
                ax.axvline(start, color="red", linestyle="--", linewidth=0.8)
                ax.axvline(end, color="red", linestyle="--", linewidth=0.8)
                for i_on, i_off in zip(light_on_idx, light_off_idx):
                    ax.axvspan(i_off, i_on, alpha=0.15, color="gray")
                ax.set_ylim(0, 100)
                ax.set_ylabel(f"{sex}\n{ani}", rotation=0, labelpad=20,
                              ha="right", va="center", fontsize=9)
                if i_ani < n_ani_fig - 1:
                    ax.tick_params(labelbottom=False)
                sns.despine(ax=ax)
            # Bottom subplot: x-axis ticks at light transitions
            all_transitions = sorted(set(light_on_idx + light_off_idx))
            t_labels = []
            for tidx in all_transitions:
                if 0 <= tidx < len(timestamps):
                    t_labels.append(timestamps[tidx].strftime("%H:%M"))
                else:
                    t_labels.append("")
            axes[-1, 0].set_xticks(all_transitions)
            axes[-1, 0].set_xticklabels(t_labels, rotation=45, ha="right",
                                         fontsize=7)
            fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.06,
                                left=0.12, right=0.99)
            self.figure_ready.emit(f"Raw data wk{week_num} ({csv_path.stem})", fig)

            step += 1
            self.progress.emit(int(step / total_steps * 100))

        # ---- PHASE 2: Bout Analysis per week ----
        # Each week may have multiple npz files (from different recording
        # sessions / cohorts).  Per-animal metrics are computed from whichever
        # npz file contains that animal, then pooled into group-level lists.
        self.log.emit("\n=== Phase 2: Bout Analysis ===")
        act_levels = get_activity_levels(p.analysis_mode)
        activity_types = list(act_levels.keys())
        prefix = "binary_" if p.analysis_mode == "binary" else "activitylevels_"
        phases_list = ["all", "dark", "light"]
        n_phases = len(phases_list)
        n_act = len(activity_types)

        for week_num in sorted(npz_files.keys()):
            self.log.emit(f"\nAnalysing week {week_num}...")

            # Create bout metric figures (one set per week, pooling all files)
            fig_dur, ax_dur = plt.subplots(n_phases, n_act, figsize=(10, 10), squeeze=False)
            fig_dur.suptitle(f"Bout Duration — Week {week_num}")
            fig_n, ax_n = plt.subplots(n_phases, n_act, figsize=(10, 10), squeeze=False)
            fig_n.suptitle(f"Bout Count — Week {week_num}")
            fig_per, ax_per = plt.subplots(n_phases, n_act, figsize=(10, 10),
                                           sharey="row", squeeze=False)
            fig_per.suptitle(f"Percent Time — Week {week_num}")

            stats_lines.append(f"\n{'='*50}\nWeek {week_num} — {prefix.rstrip('_')}\n{'='*50}\n")

            # Initialise per-group metric accumulators
            durs_by_group: dict[str, dict[str, dict[str, list[float]]]] = {}
            nums_by_group: dict[str, dict[str, dict[str, list[float]]]] = {}
            pcts_by_group: dict[str, dict[str, dict[str, list[float]]]] = {}
            for gr in groups:
                durs_by_group[gr] = {ph: {act: [] for act in activity_types} for ph in phases_list}
                nums_by_group[gr] = {ph: {act: [] for act in activity_types} for ph in phases_list}
                pcts_by_group[gr] = {ph: {act: [] for act in activity_types} for ph in phases_list}

            # Process each npz file for this week
            for npz_path in npz_files[week_num]:
                raw = np.load(npz_path, allow_pickle=True)
                activity = raw["activity"]
                anim = raw["animals"].tolist() if isinstance(raw["animals"], np.ndarray) else raw["animals"]
                lights_on = raw["light_on_indexes"].tolist()
                lights_off = raw["light_off_indexes"].tolist()

                bc = p.binning_config
                activity_bin = bin_activity(
                    activity, mode=p.analysis_mode, sleep_threshold=bc.sleep_threshold,
                    low_max=bc.low_max, medium_max=bc.medium_max, high_max=bc.high_max,
                )
                activity_bin = remove_isolated_zeros(activity_bin)
                dark_epochs, light_epochs = segment_by_phase(activity_bin, lights_on, lights_off)
                phases_data = {"all": activity_bin, "dark": dark_epochs, "light": light_epochs}

                for gr in groups:
                    gr_animals = meta.loc[meta["group_name"] == gr, "mouse"].tolist()
                    gr_indices = [anim.index(a) for a in gr_animals if a in anim]

                    for idx in gr_indices:
                        for ph_name in phases_list:
                            for act_name in activity_types:
                                bout_val = act_levels[act_name]
                                if ph_name == "all":
                                    m = compute_bout_metrics(activity_bin[idx], bout_val)
                                else:
                                    epoch = phases_data[ph_name]
                                    n_cyc = epoch.shape[-1]
                                    cyc_vals = [compute_bout_metrics(epoch[idx, :, c], bout_val) for c in range(n_cyc)]
                                    m = {
                                        "mean_duration": float(np.mean([v["mean_duration"] for v in cyc_vals])),
                                        "n_bouts": float(np.mean([v["n_bouts"] for v in cyc_vals])),
                                        "percent_time": float(np.mean([v["percent_time"] for v in cyc_vals])),
                                    }
                                durs_by_group[gr][ph_name][act_name].append(m["mean_duration"])
                                nums_by_group[gr][ph_name][act_name].append(m["n_bouts"])
                                pcts_by_group[gr][ph_name][act_name].append(m["percent_time"])

            # Now plot from the accumulated metrics
            for i_ph, ph_name in enumerate(phases_list):
                for i_act, act_name in enumerate(activity_types):
                    # Plot bars
                    for i_gr, gr in enumerate(groups):
                        vals_d = durs_by_group[gr][ph_name][act_name]
                        vals_n = nums_by_group[gr][ph_name][act_name]
                        vals_p = pcts_by_group[gr][ph_name][act_name]
                        c = colors.get(gr, "gray")
                        for ax_, vals in [(ax_dur, vals_d), (ax_n, vals_n), (ax_per, vals_p)]:
                            a = ax_[i_ph, i_act]
                            sem = np.std(vals) / np.sqrt(len(vals)) if vals else 0
                            a.bar(i_gr, np.mean(vals) if vals else 0, yerr=sem, color=c, alpha=0.5)
                            a.scatter([i_gr] * len(vals), vals, color=c, alpha=0.4, s=20, zorder=3)

                    # Labels and x-tick group names
                    if i_ph == 0:
                        ax_dur[i_ph, i_act].set_title(act_name, fontsize=10)
                        ax_n[i_ph, i_act].set_title(act_name, fontsize=10)
                        ax_per[i_ph, i_act].set_title(act_name, fontsize=10)
                    if i_act == 0:
                        ax_dur[i_ph, i_act].set_ylabel(ph_name, fontsize=10)
                        ax_n[i_ph, i_act].set_ylabel(ph_name, fontsize=10)
                        ax_per[i_ph, i_act].set_ylabel(ph_name, fontsize=10)
                    for a in [ax_dur[i_ph, i_act], ax_n[i_ph, i_act], ax_per[i_ph, i_act]]:
                        a.set_xticks(range(len(groups)))
                        a.set_xticklabels(groups, fontsize=8)
                        sns.despine(ax=a)

                    # --- Pairwise stats with FDR + sig brackets ---
                    # Bout duration (Mann-Whitney)
                    gd_dur = {gr: durs_by_group[gr][ph_name][act_name]
                              for gr in groups
                              if len(durs_by_group[gr][ph_name][act_name]) >= 2}
                    if len(gd_dur) >= 2:
                        res_dur = pairwise_fdr_tests(gd_dur, test="mannwhitney")
                        sig_dur = [(groups.index(g1), groups.index(g2), p_val)
                                   for g1, g2, p_val, is_sig in res_dur if is_sig]
                        add_sig_brackets(ax_dur[i_ph, i_act], sig_dur)
                        for g1, g2, p_val, is_sig in res_dur:
                            stats_lines.append(
                                f"{act_name} -- {ph_name} Duration ({g1} vs {g2}): "
                                f"p={p_val:.4f}{' *' if is_sig else ''}"
                            )

                    # Bout count (Mann-Whitney)
                    gd_num = {gr: nums_by_group[gr][ph_name][act_name]
                              for gr in groups
                              if len(nums_by_group[gr][ph_name][act_name]) >= 2}
                    if len(gd_num) >= 2:
                        res_num = pairwise_fdr_tests(gd_num, test="mannwhitney")
                        sig_num = [(groups.index(g1), groups.index(g2), p_val)
                                   for g1, g2, p_val, is_sig in res_num if is_sig]
                        add_sig_brackets(ax_n[i_ph, i_act], sig_num)
                        for g1, g2, p_val, is_sig in res_num:
                            stats_lines.append(
                                f"{act_name} -- {ph_name} Count ({g1} vs {g2}): "
                                f"p={p_val:.4f}{' *' if is_sig else ''}"
                            )

                    # Percent time (Welch's t-test)
                    gd_pct = {gr: pcts_by_group[gr][ph_name][act_name]
                              for gr in groups
                              if len(pcts_by_group[gr][ph_name][act_name]) >= 2}
                    if len(gd_pct) >= 2:
                        res_pct = pairwise_fdr_tests(gd_pct, test="welch")
                        sig_pct = [(groups.index(g1), groups.index(g2), p_val)
                                   for g1, g2, p_val, is_sig in res_pct if is_sig]
                        add_sig_brackets(ax_per[i_ph, i_act], sig_pct)
                        for g1, g2, p_val, is_sig in res_pct:
                            stats_lines.append(
                                f"{act_name} -- {ph_name} %Time ({g1} vs {g2}): "
                                f"p={p_val:.4f}{' *' if is_sig else ''}"
                            )

            for f in (fig_dur, fig_n, fig_per):
                f.tight_layout()
            self.figure_ready.emit(f"Bout duration wk{week_num}", fig_dur)
            self.figure_ready.emit(f"Bout count wk{week_num}", fig_n)
            self.figure_ready.emit(f"Percent time wk{week_num}", fig_per)

            step += 1
            self.progress.emit(int(step / total_steps * 100))

        # ---- PHASE 3: Per-animal control plots ----
        # ---- PHASE 4: Group comparison plots (pooled across all npz files) ----
        self.log.emit("\n=== Phase 3: Per-Animal Control Plots ===")
        group_map = {
            gr: meta.loc[meta["group_name"] == gr, "mouse"].tolist()
            for gr in groups
        }

        # Accumulate data for cross-week light/dark ratio (plot 12)
        # Structure: {group: {week: list of per-animal arrays}}
        # Each per-animal array has shape (n_days,) with one ratio per day.
        ratio_data: dict[str, dict[int, list[list[float]]]] = {gr: {} for gr in groups}
        weekly_bout_pool: dict[int, dict[str, dict[str, list[float]]]] = {}

        # Helper: resample one day to 24 hourly bins (defined once, used in Phase 4)
        def _hourly_1d(arr_1d: np.ndarray) -> np.ndarray:
            bph = max(1, len(arr_1d) // 24)
            n_h = min(24, len(arr_1d) // bph)
            return np.array([np.mean(arr_1d[h * bph:(h + 1) * bph]) for h in range(n_h)])

        for week_num in sorted(npz_files.keys()):
            # ---- Accumulators for Phase 4 group comparison plots ----
            # Filled from ALL npz files for this week, keyed by group name.
            wk_hourly: dict[str, list[np.ndarray]] = {gr: [] for gr in groups}
            wk_frag: dict[str, list[np.ndarray]] = {gr: [] for gr in groups}
            wk_cumul: dict[str, list[np.ndarray]] = {gr: [] for gr in groups}
            wk_cumul_dark: dict[str, list[np.ndarray]] = {gr: [] for gr in groups}
            wk_cumul_light: dict[str, list[np.ndarray]] = {gr: [] for gr in groups}
            # Per-day bout durations: {group: {day_idx: [dur, dur, ...]}}
            wk_bout_durs: dict[str, dict[int, list[float]]] = {
                gr: {} for gr in groups
            }
            max_n_days = 0

            # Process each npz file for this week (one per recording session)
            for npz_path in npz_files[week_num]:
                raw = np.load(npz_path, allow_pickle=True)
                activity_raw = raw["activity"]
                anim = raw["animals"].tolist() if isinstance(raw["animals"], np.ndarray) else raw["animals"]
                l_on = raw["light_on_indexes"].tolist()
                l_off = raw["light_off_indexes"].tolist()

                bc = p.binning_config
                activity_bin_w = bin_activity(
                    activity_raw, mode=p.analysis_mode, sleep_threshold=bc.sleep_threshold,
                    low_max=bc.low_max, medium_max=bc.medium_max, high_max=bc.high_max,
                )
                activity_bin_w = remove_isolated_zeros(activity_bin_w)
                dark_ep, light_ep = segment_by_phase(activity_bin_w, l_on, l_off)

                n_days_w = len(l_off) - 1
                max_n_days = max(max_n_days, n_days_w)

                # --- Phase 3: per-animal plots ---
                for i_ani, ani in enumerate(anim):
                    gr_match = meta.loc[meta["mouse"] == ani, "group_name"]
                    ani_color = colors.get(gr_match.values[0], "gray") if len(gr_match) > 0 else "gray"

                    for d in range(n_days_w):
                        s_d, e_d = l_off[d], l_off[d + 1]
                        if e_d <= s_d:
                            continue
                        day_data = activity_raw[i_ani, s_d:e_d]
                        fig_h = plot_hourly_activity_bars(
                            day_data, l_off, l_on, ani, d + 1,
                            week_num=week_num, group_color=ani_color,
                        )
                        self.figure_ready.emit(
                            f"individual_animal_plots/Hourly {ani} wk{week_num} d{d+1}", fig_h)

                    fig_ov = plot_daily_activity_overlay(
                        activity_raw[i_ani], l_off, l_on, ani, week_num,
                        group_color=ani_color)
                    self.figure_ready.emit(
                        f"individual_animal_plots/Daily overlay {ani} wk{week_num}", fig_ov)

                    fig_sv = plot_sleep_bout_violins_animal(
                        dark_ep, light_ep, i_ani, ani, week_num,
                        group_color=ani_color)
                    self.figure_ready.emit(
                        f"individual_animal_plots/Sleep violins {ani} wk{week_num}", fig_sv)

                    fig_ac = plot_actogram(
                        activity_raw[i_ani], l_off, ani, week_num)
                    self.figure_ready.emit(
                        f"individual_animal_plots/Actogram {ani} wk{week_num}", fig_ac)

                self.log.emit(f"  Control plots done for week {week_num} ({npz_path.stem})")

                # --- Accumulate per-animal data for Phase 4 ---
                day_gaps = [l_off[d + 1] - l_off[d] for d in range(n_days_w)]
                valid_gaps = [g for g in day_gaps if g > 0]
                day_len = min(valid_gaps) if valid_gaps else 0

                for gr in groups:
                    gr_animals = [a for a in group_map[gr] if a in anim]
                    for ani in gr_animals:
                        idx = anim.index(ani)

                        # 8. Hourly profile (24 values, averaged across days)
                        day_profiles: list[np.ndarray] = []
                        for d in range(n_days_w):
                            s_d, e_d = l_off[d], l_off[d + 1]
                            if e_d > s_d:
                                day_profiles.append(_hourly_1d(activity_raw[idx, s_d:e_d]))
                        if day_profiles:
                            wk_hourly[gr].append(np.mean(day_profiles, axis=0))

                        # 9. Sleep bout durations per day
                        for d in range(n_days_w):
                            n_cyc_dark = dark_ep.shape[-1]
                            n_cyc_light = light_ep.shape[-1]
                            cyc_d = min(d, n_cyc_dark - 1)
                            cyc_l = min(d, n_cyc_light - 1)
                            dark_durs = collect_bout_durations(dark_ep[idx, :, cyc_d], bout_val=0)
                            light_durs = collect_bout_durations(light_ep[idx, :, cyc_l], bout_val=0)
                            wk_bout_durs[gr].setdefault(d, {"Dark": [], "Light": []})
                            if len(dark_durs) > 0:
                                wk_bout_durs[gr][d]["Dark"].extend(dark_durs.tolist())
                            if len(light_durs) > 0:
                                wk_bout_durs[gr][d]["Light"].extend(light_durs.tolist())

                        # 10. Sleep fragmentation per day
                        frag_vals: list[float] = []
                        for d in range(n_days_w):
                            s_d, e_d = l_off[d], l_off[d + 1]
                            if e_d <= s_d:
                                frag_vals.append(0.0)
                                continue
                            day_data = activity_bin_w[idx, s_d:e_d]
                            durs = collect_bout_durations(day_data, bout_val=0)
                            total_sleep = float(np.sum(day_data == 0))
                            if total_sleep > 0 and len(durs) > 0:
                                frag_vals.append(len(durs) / total_sleep)
                            else:
                                frag_vals.append(0.0)
                        wk_frag[gr].append(np.array(frag_vals))

                        # 11. Cumulative sleep (mean across days)
                        if day_len > 0:
                            per_day_cum: list[np.ndarray] = []
                            for d in range(n_days_w):
                                s_d, e_d = l_off[d], l_off[d + 1]
                                if e_d <= s_d:
                                    continue
                                dd = activity_bin_w[idx, s_d:s_d + day_len]
                                per_day_cum.append(np.cumsum(dd == 0).astype(float))
                            if per_day_cum:
                                wk_cumul[gr].append(np.mean(per_day_cum, axis=0))

                        # 11b. Phase-specific cumulative sleep
                        n_cyc_dk = dark_ep.shape[-1]
                        n_cyc_lt = light_ep.shape[-1]
                        dark_cum_days: list[np.ndarray] = []
                        light_cum_days: list[np.ndarray] = []
                        for d in range(min(n_days_w, n_cyc_dk)):
                            dark_cum_days.append(
                                np.cumsum(dark_ep[idx, :, d] == 0).astype(float)
                            )
                        for d in range(min(n_days_w, n_cyc_lt)):
                            light_cum_days.append(
                                np.cumsum(light_ep[idx, :, d] == 0).astype(float)
                            )
                        if dark_cum_days:
                            wk_cumul_dark[gr].append(np.mean(dark_cum_days, axis=0))
                        if light_cum_days:
                            wk_cumul_light[gr].append(np.mean(light_cum_days, axis=0))

                        # 12. Light/dark ratio — one array per animal
                        ani_ratios: list[float] = []
                        for d in range(n_days_w):
                            s_d, e_d = l_off[d], l_off[d + 1]
                            if e_d <= s_d:
                                ani_ratios.append(np.nan)
                                continue
                            mid_candidates = [i for i in l_on if s_d < i < e_d]
                            if not mid_candidates:
                                ani_ratios.append(np.nan)
                                continue
                            mid = mid_candidates[0]
                            dark_mean = float(np.mean(activity_raw[idx, s_d:mid]))
                            light_mean = float(np.mean(activity_raw[idx, mid:e_d]))
                            if light_mean > 0:
                                ani_ratios.append(dark_mean / light_mean)
                            else:
                                ani_ratios.append(0.0)
                        ratio_data[gr].setdefault(week_num, []).append(ani_ratios)

            # ---- Phase 4: Generate group comparison plots from pooled data ----
            self.log.emit(f"  Group comparison plots for week {week_num} (all groups)...")
            for _dbg_gr in groups:
                self.log.emit(
                    f"    [{_dbg_gr}] hourly={len(wk_hourly[_dbg_gr])}, "
                    f"frag={len(wk_frag[_dbg_gr])}, cumul={len(wk_cumul[_dbg_gr])}, "
                    f"bout_days={len(wk_bout_durs[_dbg_gr])}"
                )

            # 8. Weekly activity profile
            gh = {gr: np.array(v) for gr, v in wk_hourly.items() if v}
            if gh:
                fig_wp = plot_weekly_activity_profile(
                    colors=colors, week_num=week_num, group_hourly=gh)
                self.figure_ready.emit(f"Activity profile wk{week_num}", fig_wp)

            # 9. Daily sleep bout violins by group
            for d in range(max_n_days):
                gbd: dict[str, dict[str, list[float]]] = {}
                for gr in groups:
                    if d in wk_bout_durs[gr]:
                        gbd[gr] = wk_bout_durs[gr][d]
                    else:
                        gbd[gr] = {"Dark": [], "Light": []}
                if any(gbd[gr]["Dark"] or gbd[gr]["Light"] for gr in gbd):
                    fig_gv, boot_gv = plot_sleep_bout_violins_group(
                        colors=colors, day_index=d, week_num=week_num,
                        group_bout_durs=gbd)
                    self.figure_ready.emit(
                        f"Group sleep violins wk{week_num} d{d+1}", fig_gv)
                    if boot_gv:
                        fig_bh = plot_bootstrap_histograms(
                            boot_gv,
                            title=f"Bootstrap — sleep bouts wk{week_num} d{d+1}",
                        )
                        if fig_bh is not None:
                            self.figure_ready.emit(
                                f"Group sleep violins wk{week_num} d{d+1} bootstrap",
                                fig_bh,
                            )

            # Pool bout durations across all days for the weekly summary violin
            wk_pool: dict[str, dict[str, list[float]]] = {}
            for gr in groups:
                wk_pool[gr] = {"Dark": [], "Light": []}
                for d_data in wk_bout_durs[gr].values():
                    wk_pool[gr]["Dark"].extend(d_data.get("Dark", []))
                    wk_pool[gr]["Light"].extend(d_data.get("Light", []))
            weekly_bout_pool[week_num] = wk_pool

            # 10. Sleep fragmentation
            gf = {gr: np.array(v) for gr, v in wk_frag.items() if v}
            if gf:
                fig_sf = plot_sleep_fragmentation(
                    colors=colors, week_num=week_num, group_frag=gf)
                self.figure_ready.emit(f"Sleep frag wk{week_num}", fig_sf)

            # 11a. Cumulative sleep 24h (descriptive, x-axis in hours)
            gc = {gr: np.array(v) for gr, v in wk_cumul.items() if v}
            if gc:
                fig_cs24 = plot_cumulative_sleep_24h(gc, colors, week_num)
                self.figure_ready.emit(f"Cumul sleep 24h wk{week_num}", fig_cs24)

            # 11b. Cumulative sleep by phase (dark/light panels)
            gc_dark = {gr: np.array(v) for gr, v in wk_cumul_dark.items() if v}
            gc_light = {gr: np.array(v) for gr, v in wk_cumul_light.items() if v}
            if gc_dark or gc_light:
                fig_csph = plot_cumulative_sleep_phases(
                    gc_dark, gc_light, colors, week_num)
                self.figure_ready.emit(f"Cumul sleep phases wk{week_num}", fig_csph)

            # 11c. Cumulative sleep stats (endpoint comparison)
            if gc:
                fig_cs = plot_cumulative_sleep(
                    colors=colors, week_num=week_num, group_cumul=gc)
                self.figure_ready.emit(f"Cumul sleep stats wk{week_num}", fig_cs)

        # 12. Light/dark activity ratio (cross-week)
        if any(ratio_data[gr] for gr in groups):
            fig_ratio = plot_light_dark_ratio(
                ratio_data, colors, sorted(npz_files.keys()))
            self.figure_ready.emit("L D ratio all weeks", fig_ratio)

        # 13. Weekly sleep bout violins (cross-week summary)
        if weekly_bout_pool:
            fig_wbv, boot_wbv = plot_weekly_sleep_bout_violins(
                weekly_bout_pool, colors, sorted(npz_files.keys()))
            self.figure_ready.emit("Weekly sleep violins", fig_wbv)
            if boot_wbv:
                fig_bh_w = plot_bootstrap_histograms(
                    boot_wbv,
                    title="Bootstrap — weekly sleep bouts",
                )
                if fig_bh_w is not None:
                    self.figure_ready.emit("Weekly sleep violins bootstrap", fig_bh_w)

        # Emit stats
        self.stats_text.emit("\n".join(stats_lines))
        self.progress.emit(100)
        self.log.emit("\nKir2.1 pipeline complete.")
