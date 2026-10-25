# COMPASS Analysis

A Python toolkit for analyzing home-cage activity data from [COMPASS](https://pmc.ncbi.nlm.nih.gov/articles/PMC5140024/) actimeter boxes (PIR-based motion sensors). Designed for neuroscience experiments studying the effects of neural manipulations on activity and sleep-wake behavior in rodents.

COMPASS (**C**ontinuous **O**pen **M**ouse **P**henotyping of **A**ctivity and **S**leep **S**tatus) is a non-invasive system that uses passive infrared sensors to monitor home-cage activity. This toolkit provides end-to-end analysis pipelines and a PyQt6 GUI for processing raw COMPASS recordings into publication-ready figures and statistics.

## Supported Experiment Type

| Experiment | Duration | Design |
|---|---|---|
| **Kir2.1** (Constitutive Inhibition) | Multiple weeks | Control vs experimental groups, referenced to SNI surgery date |

## Features

- PyQt6 GUI with per-file time correction, week selection, and dynamic group colors
- Parse COMPASS CSV files with configurable calibration (activity cap, noise threshold)
- Detect and validate light/dark cycle transitions
- Bin activity into categorical levels (passive, low, medium, high)
- Detect and quantify activity/inactivity bouts
- Statistical comparisons (Welch's t-test, Mann-Whitney U, FDR correction, bootstrap)
- Publication-ready figures (Adobe Illustrator-compatible PDFs)

## Installation

```bash
git clone https://github.com/raqueladaia/mouse_actimetry_COMPASS.git
cd mouse_actimetry_COMPASS
pip install -e ".[gui,dev]"
```

To install without the GUI (core library only):

```bash
pip install -e .
```

## Quick Start

### GUI

```bash
python run.py
# or
python -m compass.gui
```

## Using the GUI

### 1. Select files

- **Data folder** — Browse to the folder containing your COMPASS CSV files. The GUI auto-detects weeks by parsing filenames (expected format: `{date}_{hour}_{week}_{groupids}.csv`, e.g. `240213_1806_week1_gr1+gr2.csv`). It also populates the per-file time offset table.
- **Metadata file** — Browse to your `info_mice.xlsx` Excel file. The GUI reads the `group_name` column and creates a color picker button for each group.
- **Output folder** — Where figures (PDF) and processed data (.npz) will be saved.

### 2. Configure weeks and time offsets

- **Week Selection** — After selecting the data folder, checkboxes appear for each detected week. Uncheck any weeks you want to skip.
- **Per-File Time Offsets** — In case the COMPASS PC has the wrong time configured, you can enter a correction offset (in minutes) for each CSV file. The offset is applied to all timestamps before detecting light/dark cycle transitions. Each file can have a different offset.

### 3. Set parameters

- **Lights off / Lights on time** — The light schedule times (default: off at 09:30, on at 21:30).
- **Analysis mode** — `levels` bins activity into 4 categories (passive, low, medium, high); `binary` uses only active/inactive.
- **Binning config** — Optionally load a custom JSON config to change activity binning thresholds and calibration settings (activity cap, noise threshold). If left empty, defaults are used. See `config_files/default_config.json` for the format.

### 4. Customize group colors

After loading the metadata file, one color button appears per group. Click a button to open a color picker and override the default. These colors are used consistently across all plots.

### 5. Run the pipeline

Click **Run Pipeline**. The progress bar, log panel, and figure viewer update in real time. The pipeline runs three phases:

1. **Data collection** — Loads each CSV, applies time correction, detects light cycles, trims to the analysis window, and saves `.npz` files. Generates a raw data overview figure per file.
2. **Bout analysis** — Bins activity, detects bouts, and computes duration/count/percent-time metrics per group. Generates bout metric grid plots with statistical comparisons.
3. **Per-animal and group comparison plots** — Generates all remaining figures: hourly activity, daily overlays, actograms, sleep violins, cumulative sleep, fragmentation, and light/dark ratio.

### 6. Review results

- **Figure viewer** — Browse generated figures in the right panel. Figures are also saved as PDFs in the output folder. Per-animal plots go into an `individual_animal_plots/` subfolder.
- **Export Statistics** — After the pipeline completes, click **Export Statistics** to save the statistical results as a text file.

### Python API

```python
from compass.io.csv_parser import load_compass_csv
from compass.io.metadata import load_metadata
from compass.processing.light_cycles import detect_light_cycles
from compass.processing.activity import bin_activity
from compass.processing.bouts import collect_bout_durations

# Load and process data
raw_data, timestamps, warnings = load_compass_csv(
    "path/to/data.csv",
    time_offset_min=40,
    activity_cap=100,       # clip values above this (default 100)
    noise_threshold=0,      # zero values below this (default 0 = disabled)
)
metadata = load_metadata("path/to/info_mice.xlsx")
light_on, light_off = detect_light_cycles(timestamps, lights_off_time="09:30", lights_on_time="21:30")
binned = bin_activity(raw_data)
durations = collect_bout_durations(binned[0], bout_val=0)
```

## Configuration

Analysis parameters are controlled via a JSON config file. The default is at `config_files/default_config.json`:

```json
{
  "activity_binning": {
    "sleep_threshold": 1,
    "boundaries": { "low_max": 20, "medium_max": 50, "high_max": 100 }
  },
  "calibration": {
    "activity_cap": 100,
    "noise_threshold": 0
  }
}
```

- **activity_cap**: Raw values above this are clipped (a warning is emitted). Default: 100.
- **noise_threshold**: Raw values below this are zeroed. Default: 0 (disabled).

A custom config file can be selected in the GUI under Parameters.

## Project Structure

```
mouse_actimetry_COMPASS/
├── compass/              # Main package
│   ├── io/               # File I/O (CSV, Excel, .npz, config)
│   ├── processing/       # Data transforms (binning, bout detection)
│   ├── analysis/         # Statistical analysis
│   ├── plotting/         # Figure generation
│   ├── gui/              # PyQt6 GUI
│   └── experiments/      # Experiment-specific pipelines
├── config_files/         # Default configuration
├── tests/                # Unit tests
├── CLAUDE.md             # Development guide for Claude Code
└── INFORMATION.md        # Detailed scientific & technical docs
```

## Data Requirements

### COMPASS CSV files

Raw actimeter recordings with datetime-indexed rows and `PIR{N}` columns (one per box). Files must follow the naming convention `{date}_{hour}_{week}_{groupids}.csv`:

| Component | Description | Example |
|-----------|-------------|---------|
| date | Recording start date (YYMMDD) | `240213` |
| hour | Recording start time (HHMM) | `1806` |
| week | Week identifier | `week1`, `week2` |
| groupids | Group ID(s) from metadata, joined by `+` | `gr1`, `gr1+gr2` |

Example filenames: `240213_1806_week1_gr1+gr2.csv`, `260205_1552_week1_gr3.csv`

Multiple CSV files can share the same week number (different cohorts recorded at different times).

### Metadata file (`info_mice.xlsx`)

An Excel file (`.xlsx`) with one row per animal. The GUI reads this file to determine group assignments, box mappings, and the SNI surgery reference date. Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `mouse` | int | Unique animal ID |
| `group_id` | str | Identifier linking the animal to CSV filenames (must match the `groupids` component in the filename) |
| `group_name` | str | Human-readable group label (e.g., `control`, `Kir2.1`). Used for plot legends and color assignment |
| `sex` | str | `M` or `F` |
| `actimeter_box` | int | Box number (1-based). Maps to `PIR{N}` in the raw CSV — box 1 = `header` column, box N = `PIR{N-1}` column |
| `sni_date` | datetime | Date of SNI surgery. Used to determine the analysis window |

Example:

| mouse | group_id | group_name | sex | actimeter_box | sni_date |
|-------|----------|------------|-----|---------------|----------|
| 101 | gr1 | control | M | 1 | 2024-02-01 |
| 102 | gr1 | control | F | 2 | 2024-02-01 |
| 201 | gr2 | Kir2.1 | M | 3 | 2024-02-01 |
| 202 | gr2 | Kir2.1 | F | 4 | 2024-02-01 |

See [INFORMATION.md](INFORMATION.md) for additional technical details on the data format.

## Dependencies

- Python >= 3.10
- **Core**: numpy, pandas, scipy, matplotlib, seaborn, statsmodels, openpyxl
- **GUI**: PyQt6
- **Dev**: pytest, pytest-cov

## References

Brown, L. A., Hasan, S., Foster, R. G., & Peirson, S. N. (2017). COMPASS: Continuous Open Mouse Phenotyping of Activity and Sleep Status. *Wellcome Open Research*, 1, 2. https://doi.org/10.12688/wellcomeopenres.9892.2

## License

MIT
