"""Per-animal control plots for the Kir2.1 experiment.

Plot types
----------
1. Hourly activity bar plots (polar + Cartesian) — one figure per day per animal
2. Daily activity overlay line plots (polar + Cartesian) — one figure per animal
3. Sleep bout duration violin plots — one figure per animal
4. Single-plotted actogram — one figure per animal
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from matplotlib.figure import Figure

from compass.processing.light_cycles import NORMAL_HALF_CYCLE_BINS


# Number of bins in one full 24 h cycle (~10 s each)
_BINS_PER_DAY = 2 * NORMAL_HALF_CYCLE_BINS


def _setup_polar(ax: plt.Axes) -> None:
    """Apply standard polar conventions: 0 h at top, clockwise."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


def _sequential_cmap(base_color: str) -> LinearSegmentedColormap:
    """Create a white-to-color sequential colormap for day progression."""
    rgba = to_rgba(base_color)
    return LinearSegmentedColormap.from_list("custom_seq", [(1, 1, 1, 1), rgba])


# ------------------------------------------------------------------
# 1.  Hourly activity bar plots
# ------------------------------------------------------------------

def plot_hourly_activity_bars(
    activity_1d: np.ndarray,
    lights_off_idx: list[int],
    lights_on_idx: list[int],
    animal_id: int | str,
    day_index: int,
    week_num: int = 0,
    group_color: str = "steelblue",
) -> Figure:
    """Polar + Cartesian bar plot of mean hourly activity for one day.

    Parameters
    ----------
    activity_1d : np.ndarray
        Raw (0-100) activity for one animal, one day (~8648 bins).
    lights_off_idx, lights_on_idx : list[int]
        Transition indices *relative to the day slice*.
    animal_id : int or str
    day_index : int
    week_num : int
    group_color : str

    Returns
    -------
    Figure
    """
    n = len(activity_1d)
    bins_per_hour = max(1, n // 24)
    n_hours = min(24, n // bins_per_hour)

    hourly = np.array([
        np.mean(activity_1d[h * bins_per_hour:(h + 1) * bins_per_hour])
        for h in range(n_hours)
    ])

    angles = np.linspace(0, 2 * np.pi, n_hours, endpoint=False)
    width = 2 * np.pi / n_hours

    fig, (ax_polar, ax_cart) = plt.subplots(
        1, 2,
        figsize=(12, 5),
        subplot_kw={"projection": "polar"},
    )
    # Recreate ax_cart as Cartesian
    fig.delaxes(ax_cart)
    ax_cart = fig.add_subplot(1, 2, 2)

    # --- Polar ---
    _setup_polar(ax_polar)
    ax_polar.bar(angles, hourly, width=width * 0.9, alpha=0.7,
                 color=group_color, edgecolor="white", linewidth=0.5)
    # Shade dark phase (top half = 0-12 h from lights off)
    ax_polar.fill_between(
        np.linspace(0, np.pi, 100), 0, ax_polar.get_ylim()[1],
        alpha=0.08, color="gray",
    )
    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax_polar.set_xticklabels([str(h) for h in range(24)], fontsize=7)
    ax_polar.set_title(f"Animal {animal_id} — day {day_index}", fontsize=10, pad=14)

    # --- Cartesian ---
    hours = np.arange(n_hours)
    ax_cart.bar(hours, hourly, color=group_color, alpha=0.7, edgecolor="white",
                linewidth=0.5)
    # Shade dark phase (first 12 h from lights-off = hours 0-11)
    ax_cart.axvspan(-0.5, 11.5, alpha=0.10, color="gray", label="dark")
    ax_cart.set_xlim(-0.5, n_hours - 0.5)
    ax_cart.set_xlabel("Hour from lights off")
    ax_cart.set_ylabel("Mean activity (0-100)")
    ax_cart.set_ylim(0, 100)
    ax_cart.set_title(f"Animal {animal_id} — day {day_index} (wk{week_num})",
                      fontsize=10)
    sns.despine(ax=ax_cart)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 2.  Daily activity overlay line plots
# ------------------------------------------------------------------

def plot_daily_activity_overlay(
    activity_1d: np.ndarray,
    lights_off_idx: list[int],
    lights_on_idx: list[int],
    animal_id: int | str,
    week_num: int,
    group_color: str = "steelblue",
) -> Figure:
    """Polar + Cartesian overlay of 24 h activity profiles across days.

    Parameters
    ----------
    activity_1d : np.ndarray
        Raw activity for one animal over one week (all timepoints).
    lights_off_idx, lights_on_idx : list[int]
        Light-cycle transition indices (within the trimmed data).
    animal_id : int or str
    week_num : int
    group_color : str

    Returns
    -------
    Figure
    """
    # Split into days using lights_off boundaries
    day_slices: list[np.ndarray] = []
    for i in range(len(lights_off_idx) - 1):
        s, e = lights_off_idx[i], lights_off_idx[i + 1]
        if e > s:
            day_slices.append(activity_1d[s:e])

    n_days = len(day_slices)
    if n_days == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No complete days", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    cmap = _sequential_cmap(group_color)
    norm_vals = np.linspace(0.3, 1.0, n_days)

    fig = plt.figure(figsize=(12, 5))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_cart = fig.add_subplot(1, 2, 2)

    _setup_polar(ax_polar)

    for d_i, day_data in enumerate(day_slices):
        color = cmap(norm_vals[d_i])
        # Resample to 24 bins (one per hour) for cleaner overlay
        bins_per_hour = max(1, len(day_data) // 24)
        n_hours = min(24, len(day_data) // bins_per_hour)
        hourly = np.array([
            np.mean(day_data[h * bins_per_hour:(h + 1) * bins_per_hour])
            for h in range(n_hours)
        ])
        angles = np.linspace(0, 2 * np.pi, n_hours, endpoint=False)
        hours = np.arange(n_hours)

        # Close the polar loop
        angles_closed = np.append(angles, angles[0])
        hourly_closed = np.append(hourly, hourly[0])

        ax_polar.plot(angles_closed, hourly_closed, color=color, alpha=0.7,
                      linewidth=1.0)
        ax_cart.plot(hours, hourly, color=color, alpha=0.7, linewidth=1.0,
                     marker="o", markersize=3)

    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax_polar.set_xticklabels([str(h) for h in range(24)], fontsize=7)

    ax_cart.axvspan(-0.5, 11.5, alpha=0.10, color="gray")
    ax_cart.set_xlim(-0.5, 23.5)
    ax_cart.set_xlabel("Hour from lights off")
    ax_cart.set_ylabel("Mean activity (0-100)")
    sns.despine(ax=ax_cart)

    fig.suptitle(f"Animal {animal_id} — week {week_num}  (days 1-{n_days})",
                 fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 3.  Sleep bout duration violin plots (per animal)
# ------------------------------------------------------------------

def plot_sleep_bout_violins_animal(
    dark_epochs: np.ndarray,
    light_epochs: np.ndarray,
    animal_idx: int,
    animal_id: int | str,
    week_num: int,
    group_color: str = "steelblue",
) -> Figure:
    """Violin plots of sleep (activity=0) bout durations per day, per phase.

    Parameters
    ----------
    dark_epochs : np.ndarray
        Shape ``(n_animals, epoch_len, n_dark_cycles)``.
    light_epochs : np.ndarray
        Shape ``(n_animals, epoch_len, n_light_cycles)``.
    animal_idx : int
        Index into the animal axis.
    animal_id : int or str
    week_num : int
    group_color : str

    Returns
    -------
    Figure
    """
    from compass.processing.bouts import collect_bout_durations

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, epochs, phase_name in [
        (axes[0], dark_epochs, "Dark"),
        (axes[1], light_epochs, "Light"),
    ]:
        n_cyc = epochs.shape[-1]
        data_list: list[list[float]] = []
        positions: list[int] = []
        for c in range(n_cyc):
            durs = collect_bout_durations(epochs[animal_idx, :, c], bout_val=0)
            data_list.append(durs.tolist() if len(durs) > 0 else [0])
            positions.append(c + 1)

        if data_list:
            parts = ax.violinplot(data_list, positions=positions, showmedians=True)
            for pc in parts.get("bodies", []):
                pc.set_facecolor(group_color)
                pc.set_alpha(0.5)
        ax.set_title(f"{phase_name} phase", fontsize=10)
        ax.set_xlabel("Day (cycle)")
        ax.set_xticks(positions)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Sleep bout duration (bins)")
    fig.suptitle(f"Animal {animal_id} — week {week_num}", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 4.  Single-plotted actogram
# ------------------------------------------------------------------

def plot_actogram(
    activity_1d: np.ndarray,
    lights_off_idx: list[int],
    animal_id: int | str,
    week_num: int,
    bins_per_day: int | None = None,
) -> Figure:
    """Single-plotted actogram: each row shows one 24 h day.

    Parameters
    ----------
    activity_1d : np.ndarray
        Raw (0-100) activity for one animal over the whole trimmed recording.
    lights_off_idx : list[int]
        Indices of lights-off transitions.
    animal_id : int or str
    week_num : int
    bins_per_day : int, optional
        Defaults to 2 * NORMAL_HALF_CYCLE_BINS.

    Returns
    -------
    Figure
    """
    if bins_per_day is None:
        bins_per_day = _BINS_PER_DAY

    # Chop into days using lights-off indices
    day_slices: list[np.ndarray] = []
    for i in range(len(lights_off_idx) - 1):
        s, e = lights_off_idx[i], lights_off_idx[i + 1]
        chunk = activity_1d[s:e]
        # Pad / trim to standard day length
        if len(chunk) < bins_per_day:
            chunk = np.pad(chunk, (0, bins_per_day - len(chunk)))
        else:
            chunk = chunk[:bins_per_day]
        day_slices.append(chunk)

    n_days = len(day_slices)
    if n_days < 1:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "Not enough days for actogram", ha="center",
                va="center", transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(n_days, 1, figsize=(10, max(3, 2 * n_days)),
                             sharex=True, squeeze=False)

    x24 = np.arange(bins_per_day)
    half = bins_per_day // 2  # dark-phase length

    for r in range(n_days):
        ax = axes[r, 0]
        ax.fill_between(x24, 0, day_slices[r], color="black", linewidth=0)
        # Dark phase shading (first ~12 h from lights off)
        ax.axvspan(0, half, alpha=0.08, color="gray")
        ax.set_ylim(0, 100)
        ax.set_yticks([])
        ax.set_ylabel(f"d{r + 1}", rotation=0, labelpad=20, ha="right",
                      va="center", fontsize=8)
        sns.despine(ax=ax, left=True)

    axes[-1, 0].set_xlabel("Bins (24 h)")
    fig.suptitle(f"Actogram — Animal {animal_id}, week {week_num}", fontsize=11)
    fig.tight_layout()
    return fig
