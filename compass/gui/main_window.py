"""Main application window for COMPASS Analysis GUI."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QSplitter,
    QScrollArea,
    QMessageBox,
    QColorDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

from compass.gui.figure_viewer import FigureViewer
from compass.gui.workers import PipelineParams, Kir21Worker
from compass.io.config import (
    ActivityBinningConfig,
    CalibrationConfig,
    CompassConfig,
    DEFAULT_BINNING_CONFIG,
    DEFAULT_CALIBRATION_CONFIG,
    DEFAULT_CONFIG,
    load_config,
)
from compass.io.csv_parser import scan_csv_folder, CSVFileInfo
from compass.plotting.style import COLORS_KIR2_1, DEFAULT_GROUP_PALETTE


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("COMPASS Analysis")
        self.setMinimumSize(900, 700)

        self._worker: Kir21Worker | None = None
        self._stats_text: str = ""
        self._detected_weeks: dict[int, list[CSVFileInfo]] = {}
        self._week_checkboxes: list[QCheckBox] = []
        self._group_color_buttons: list[tuple[str, QPushButton]] = []  # (group_name, btn)
        self._binning_config: ActivityBinningConfig = DEFAULT_BINNING_CONFIG
        self._calibration_config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter)

        # --- Top panel: config ---
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(6, 6, 6, 6)

        # File paths
        files_box = QGroupBox("Files")
        files_form = QFormLayout(files_box)

        self.edit_data_folder = QLineEdit()
        self.edit_data_folder.editingFinished.connect(self._on_data_folder_edited)
        btn_data = QPushButton("Browse...")
        btn_data.clicked.connect(self._browse_data_folder)
        row_data = QHBoxLayout()
        row_data.addWidget(self.edit_data_folder)
        row_data.addWidget(btn_data)
        files_form.addRow("Data folder:", row_data)

        self.edit_metadata = QLineEdit()
        self.edit_metadata.editingFinished.connect(self._on_metadata_edited)
        self.btn_meta = QPushButton("Browse...")
        self.btn_meta.clicked.connect(self._browse_metadata_file)
        row_meta = QHBoxLayout()
        row_meta.addWidget(self.edit_metadata)
        row_meta.addWidget(self.btn_meta)
        files_form.addRow("Metadata file:", row_meta)

        self.edit_output = QLineEdit()
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(lambda: self._browse_folder(self.edit_output))
        row_out = QHBoxLayout()
        row_out.addWidget(self.edit_output)
        row_out.addWidget(btn_out)
        files_form.addRow("Output folder:", row_out)

        config_layout.addWidget(files_box)

        # --- Week selection (Kir2.1 only) ---
        self.weeks_box = QGroupBox("Week Selection")
        self._weeks_layout = QHBoxLayout(self.weeks_box)
        self._weeks_layout.setContentsMargins(6, 6, 6, 6)
        self.label_no_weeks = QLabel("Select a data folder to auto-detect weeks.")
        self.label_no_weeks.setStyleSheet("color: gray; font-style: italic;")
        self._weeks_layout.addWidget(self.label_no_weeks)
        self._weeks_layout.addStretch()
        config_layout.addWidget(self.weeks_box)

        # --- Per-file time offsets (Kir2.1 only) ---
        self.offsets_box = QGroupBox("Per-File Time Offsets (minutes)")
        offsets_layout = QVBoxLayout(self.offsets_box)

        self.offsets_table = QTableWidget(0, 2)
        self.offsets_table.setHorizontalHeaderLabels(["CSV File", "Offset (min)"])
        self.offsets_table.horizontalHeader().setStretchLastSection(True)
        self.offsets_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.offsets_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.offsets_table.setMaximumHeight(200)
        offsets_layout.addWidget(self.offsets_table)

        self.label_no_files = QLabel("Select a data folder to detect CSV files.")
        self.label_no_files.setStyleSheet("color: gray; font-style: italic;")
        offsets_layout.addWidget(self.label_no_files)

        config_layout.addWidget(self.offsets_box)

        # Parameters
        params_box = QGroupBox("Parameters")
        params_form = QFormLayout(params_box)

        self.edit_lights_off = QLineEdit("09:30:0")
        self.edit_lights_on = QLineEdit("21:30:0")
        self.label_lights_off = QLabel("Lights off time:")
        self.label_lights_on = QLabel("Lights on time:")
        params_form.addRow(self.label_lights_off, self.edit_lights_off)
        params_form.addRow(self.label_lights_on, self.edit_lights_on)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["levels", "binary"])
        params_form.addRow("Analysis mode:", self.combo_mode)

        # Binning config file picker
        self.edit_binning_config = QLineEdit()
        self.edit_binning_config.setPlaceholderText("Optional — uses defaults if empty")
        btn_binning = QPushButton("Browse...")
        btn_binning.clicked.connect(self._browse_binning_config)
        row_binning = QHBoxLayout()
        row_binning.addWidget(self.edit_binning_config)
        row_binning.addWidget(btn_binning)
        params_form.addRow("Binning config:", row_binning)

        self.label_binning_summary = QLabel(self._format_config_summary(DEFAULT_CONFIG))
        self.label_binning_summary.setStyleSheet("color: green; font-size: 11px;")
        params_form.addRow("", self.label_binning_summary)

        config_layout.addWidget(params_box)

        # --- Dynamic group colors ---
        self.colors_box = QGroupBox("Group Colors")
        self._colors_box_outer = QVBoxLayout(self.colors_box)
        self._colors_box_outer.setContentsMargins(6, 6, 6, 6)
        # Start with a placeholder container
        self._colors_container = self._make_colors_placeholder()
        self._colors_box_outer.addWidget(self._colors_container)
        config_layout.addWidget(self.colors_box)

        # Run / Export buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px 20px;")
        self.btn_run.clicked.connect(self._run_pipeline)
        btn_row.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export Statistics")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_stats)
        btn_row.addWidget(self.btn_export)
        btn_row.addStretch()
        config_layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        config_layout.addWidget(self.progress)

        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setWidget(config_widget)
        splitter.addWidget(config_scroll)

        # --- Bottom panel: log + figures ---
        bottom = QSplitter(Qt.Orientation.Horizontal)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(120)
        bottom.addWidget(self.log_box)

        self.figure_viewer = FigureViewer()
        bottom.addWidget(self.figure_viewer)

        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 2)
        splitter.addWidget(bottom)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)


    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _browse_folder(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            line_edit.setText(path)

    def _browse_data_folder(self) -> None:
        """Browse for the data folder and auto-detect CSV files / weeks."""
        path = QFileDialog.getExistingDirectory(self, "Select data folder")
        print(f"[_browse_data_folder] Dialog returned: {path!r}")
        if path:
            self.edit_data_folder.setText(path)
            self._scan_data_folder(path)

    def _on_data_folder_edited(self) -> None:
        """Called when the user finishes editing the data-folder text field."""
        path = self.edit_data_folder.text().strip()
        print(f"[_on_data_folder_edited] Text field value: {path!r}")
        if path and Path(path).is_dir():
            self._scan_data_folder(path)

    def _scan_data_folder(self, folder: str) -> None:
        """Scan the selected data folder for CSV files and populate UI."""
        print(f"[_scan_data_folder] Called with folder: {folder!r}")
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"[_scan_data_folder] Not a directory, aborting")
            return

        self._detected_weeks = scan_csv_folder(folder_path)

        # Also collect all CSV files (including ones that may not parse)
        all_csv = sorted(folder_path.glob("*.csv"))
        print(f"[_scan_data_folder] Total CSV files (all): {len(all_csv)}")
        print(f"[_scan_data_folder] Parsed weeks: {sorted(self._detected_weeks.keys())}")
        for wk, infos in sorted(self._detected_weeks.items()):
            for info in infos:
                print(f"[_scan_data_folder]   wk{wk}: {info.path.name} -> groups={info.group_ids}")

        # --- Populate week checkboxes ---
        # Clear existing checkboxes
        for cb in self._week_checkboxes:
            self._weeks_layout.removeWidget(cb)
            cb.deleteLater()
        self._week_checkboxes.clear()

        if self._detected_weeks:
            self.label_no_weeks.setVisible(False)
            for week_num in sorted(self._detected_weeks.keys()):
                n_files = len(self._detected_weeks[week_num])
                cb = QCheckBox(f"Week {week_num} ({n_files} file{'s' if n_files != 1 else ''})")
                cb.setChecked(True)
                self._weeks_layout.insertWidget(self._weeks_layout.count() - 1, cb)
                self._week_checkboxes.append(cb)
            print(f"[_scan_data_folder] Created {len(self._week_checkboxes)} week checkbox(es)")
        else:
            self.label_no_weeks.setVisible(True)
            self.label_no_weeks.setText("No parseable CSV files found in folder.")
            print("[_scan_data_folder] No weeks detected — showing placeholder label")

        # --- Populate per-file offset table ---
        self.offsets_table.setRowCount(len(all_csv))
        if all_csv:
            self.label_no_files.setVisible(False)
            for row, csv_path in enumerate(all_csv):
                # Filename (read-only)
                name_item = QTableWidgetItem(csv_path.name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.offsets_table.setItem(row, 0, name_item)
                # Offset (editable, default 0)
                offset_item = QTableWidgetItem("0")
                self.offsets_table.setItem(row, 1, offset_item)
            print(f"[_scan_data_folder] Populated offset table with {len(all_csv)} file(s)")
        else:
            self.label_no_files.setVisible(True)
            self.label_no_files.setText("No CSV files found in folder.")
            print("[_scan_data_folder] No CSV files found at all")

    def _browse_file(self, line_edit: QLineEdit, filt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if path:
            line_edit.setText(path)

    def _browse_metadata_file(self) -> None:
        """Browse for the metadata Excel file and detect groups."""
        print("[DEBUG] Metadata browse callback fired")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata file", "", "Excel files (*.xlsx *.xls)"
        )
        print(f"[DEBUG] File dialog returned: {path!r}")
        if path:
            self.edit_metadata.setText(path)
            self._detect_groups_from_metadata(path)
        else:
            print("[DEBUG] No file selected (dialog cancelled)")

    def _on_metadata_edited(self) -> None:
        """Called when the user finishes editing the metadata path field."""
        path = self.edit_metadata.text().strip()
        print(f"[DEBUG] Metadata editingFinished fired, path={path!r}")
        if path and Path(path).is_file():
            self._detect_groups_from_metadata(path)
        else:
            print(f"[DEBUG] Skipping: path empty or not a file")

    def _detect_groups_from_metadata(self, filepath: str) -> None:
        """Read the metadata Excel file and populate group color buttons."""
        import pandas as pd

        print(f"[DEBUG] Reading metadata from: {filepath}")

        try:
            df = pd.read_excel(filepath, engine="openpyxl")
        except Exception as exc:
            msg = f"Failed to read metadata: {exc}"
            print(f"[DEBUG] EXCEL READ FAILED: {msg}")
            self._show_group_colors_error(msg)
            return

        print(f"[DEBUG] Excel loaded OK. Shape: {df.shape}, Columns: {list(df.columns)}")

        candidates = ["group_name", "group", "Group_name", "Group"]

        group_col = None
        for col in candidates:
            if col in df.columns:
                group_col = col
                break

        if group_col is None:
            msg = (f"No group column found. Tried: {candidates}. "
                   f"Available: {list(df.columns)}")
            print(f"[DEBUG] COLUMN NOT FOUND: {msg}")
            self._show_group_colors_error(msg)
            return

        group_names = list(df[group_col].dropna().unique())
        print(f"[DEBUG] Found group_names: {group_names} (from column {group_col!r})")
        self._populate_group_colors(group_names)

    def _show_group_colors_error(self, msg: str) -> None:
        """Replace the group colors container with an error message."""
        self._colors_box_outer.removeWidget(self._colors_container)
        self._colors_container.setParent(None)
        self._group_color_buttons.clear()
        self._colors_container = self._make_colors_placeholder(msg)
        self._colors_box_outer.addWidget(self._colors_container)

    def _make_colors_placeholder(self, text: str = "Select a metadata file to detect groups.") -> QWidget:
        """Create a placeholder widget for the Group Colors section."""
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(text)
        label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(label)
        layout.addStretch()
        return w

    def _populate_group_colors(self, group_names: list[str]) -> None:
        """Create one color picker button per group name.

        Completely replaces the container widget inside colors_box each time,
        avoiding any dynamic layout insertion issues.
        """
        print(f"[DEBUG] _populate_group_colors called with: {group_names}")

        # Remove the old container entirely
        self._colors_box_outer.removeWidget(self._colors_container)
        self._colors_container.setParent(None)
        self._group_color_buttons.clear()

        if not group_names:
            self._colors_container = self._make_colors_placeholder(
                "No groups found in metadata."
            )
            self._colors_box_outer.addWidget(self._colors_container)
            print("[DEBUG] No groups — showing placeholder label")
            return

        # Build a fresh container with one button per group
        self._colors_container = QWidget()
        row = QHBoxLayout(self._colors_container)
        row.setContentsMargins(0, 0, 0, 0)

        for i, name in enumerate(group_names):
            default_color = DEFAULT_GROUP_PALETTE[i % len(DEFAULT_GROUP_PALETTE)]
            btn = QPushButton(str(name))
            btn.setProperty("chosen_color", default_color)
            btn.setStyleSheet(f"background-color: {default_color};")
            btn.setMinimumWidth(80)
            btn.clicked.connect(lambda checked, b=btn: self._pick_group_color(b))
            row.addWidget(btn)
            self._group_color_buttons.append((str(name), btn))
            print(f"[DEBUG] Creating color button for: {name} (color={default_color})")

        row.addStretch()
        self._colors_box_outer.addWidget(self._colors_container)

        print(f"[DEBUG] Total buttons created: {len(self._group_color_buttons)}")

    def _pick_group_color(self, btn: QPushButton) -> None:
        """Open a color dialog for a group color button."""
        color = QColorDialog.getColor()
        if color.isValid():
            btn.setStyleSheet(f"background-color: {color.name()};")
            btn.setProperty("chosen_color", color.name())

    def _format_config_summary(self, cfg: CompassConfig) -> str:
        """Return a human-readable summary of binning and calibration config."""
        b = cfg.binning
        c = cfg.calibration
        return (
            f"sleep<{b.sleep_threshold} | "
            f"low: [{b.sleep_threshold}-{b.low_max}] | "
            f"med: ({b.low_max}-{b.medium_max}] | "
            f"high: ({b.medium_max}-{b.high_max}] | "
            f"cap: {c.activity_cap} | noise: {c.noise_threshold}"
        )

    def _browse_binning_config(self) -> None:
        """Open a file dialog to select a binning config JSON file."""
        start_dir = str(Path(__file__).resolve().parents[2] / "config_files")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select binning config", start_dir, "JSON files (*.json)"
        )
        if path:
            self.edit_binning_config.setText(path)
            self._load_binning_config(path)

    def _load_binning_config(self, filepath: str) -> None:
        """Validate and load a config file, updating the summary label."""
        try:
            cfg = load_config(filepath)
            self._binning_config = cfg.binning
            self._calibration_config = cfg.calibration
            self.label_binning_summary.setText(self._format_config_summary(cfg))
            self.label_binning_summary.setStyleSheet("color: green; font-size: 11px;")
        except Exception as exc:
            self._binning_config = DEFAULT_BINNING_CONFIG
            self._calibration_config = DEFAULT_CALIBRATION_CONFIG
            self.label_binning_summary.setText(
                f"Invalid config — using defaults. ({exc})"
            )
            self.label_binning_summary.setStyleSheet("color: gray; font-size: 11px;")
            QMessageBox.warning(
                self, "Config Error",
                f"Failed to load config:\n{exc}\n\nUsing default values.",
            )

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _get_selected_weeks(self) -> list[int]:
        """Return the list of selected week numbers from checkboxes."""
        selected: list[int] = []
        for cb in self._week_checkboxes:
            if cb.isChecked():
                # Parse week number from label like "Week 1 (2 files)"
                text = cb.text()
                try:
                    week_num = int(text.split()[1])
                    selected.append(week_num)
                except (IndexError, ValueError):
                    pass
        return selected

    def _get_file_offsets(self) -> dict[str, float]:
        """Read per-file offsets from the table widget."""
        offsets: dict[str, float] = {}
        for row in range(self.offsets_table.rowCount()):
            name_item = self.offsets_table.item(row, 0)
            offset_item = self.offsets_table.item(row, 1)
            if name_item and offset_item:
                try:
                    offsets[name_item.text()] = float(offset_item.text())
                except ValueError:
                    offsets[name_item.text()] = 0.0
        return offsets

    def _build_params(self) -> PipelineParams:
        # Collect per-file offsets from the table
        file_offsets = self._get_file_offsets()

        colors: dict[str, str] = {}
        for group_name, btn in self._group_color_buttons:
            c = btn.property("chosen_color")
            if c:
                colors[group_name] = c

        # Load binning config from file if provided, otherwise use stored value
        binning_path = self.edit_binning_config.text().strip()
        if binning_path:
            self._load_binning_config(binning_path)

        return PipelineParams(
            data_folder=self.edit_data_folder.text().strip(),
            metadata_file=self.edit_metadata.text().strip(),
            output_folder=self.edit_output.text().strip(),
            file_offsets=file_offsets,
            lights_off_time=self.edit_lights_off.text().strip(),
            lights_on_time=self.edit_lights_on.text().strip(),
            selected_weeks=self._get_selected_weeks(),
            analysis_mode=self.combo_mode.currentText(),
            binning_config=self._binning_config,
            calibration_config=self._calibration_config,
            colors=colors,
        )

    def _validate(self, params: PipelineParams) -> str | None:
        """Return an error message or None if valid."""
        if not params.data_folder or not Path(params.data_folder).is_dir():
            return "Please select a valid data folder."
        if not params.metadata_file or not Path(params.metadata_file).is_file():
            return "Please select a valid metadata file."
        if not params.selected_weeks:
            return "Please select at least one week to analyze."
        if not params.output_folder:
            return "Please select an output folder."
        return None

    def _run_pipeline(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A pipeline is already running.")
            return

        params = self._build_params()
        err = self._validate(params)
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        self.log_box.clear()
        self.figure_viewer.clear()
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._stats_text = ""

        self._worker = Kir21Worker(params)

        self._worker.progress.connect(self.progress.setValue)
        self._worker.log.connect(self._append_log)
        self._worker.figure_ready.connect(self.figure_viewer.add_figure)
        self._worker.stats_text.connect(self._on_stats)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.finished_err.connect(self._on_error)
        self._worker.start()

    def _append_log(self, text: str) -> None:
        self.log_box.append(text)

    def _on_stats(self, text: str) -> None:
        self._stats_text = text
        self.btn_export.setEnabled(bool(text))

    def _on_finished(self) -> None:
        self.btn_run.setEnabled(True)
        self._append_log("\nDone.")

    def _on_error(self, tb: str) -> None:
        self.btn_run.setEnabled(True)
        self._append_log(f"\nERROR:\n{tb}")
        QMessageBox.critical(self, "Pipeline Error", "The pipeline failed. See log for details.")

    def _export_stats(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save statistics", "", "Text files (*.txt)")
        if path:
            Path(path).write_text(self._stats_text, encoding="utf-8")
            self._append_log(f"Statistics saved to {path}")
