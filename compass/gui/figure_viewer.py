"""Widget for previewing and exporting matplotlib figures."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QFileDialog,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class FigureViewer(QWidget):
    """Tabbed viewer for matplotlib figures with PDF export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figures: dict[str, Figure] = {}

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        btn_row = QHBoxLayout()
        self.btn_export_all = QPushButton("Export All as PDF")
        self.btn_export_all.clicked.connect(self._export_all)
        self.btn_export_all.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_export_all)
        layout.addLayout(btn_row)

    def clear(self) -> None:
        """Remove all figure tabs."""
        self.tabs.clear()
        self._figures.clear()
        self.btn_export_all.setEnabled(False)

    def add_figure(self, name: str, fig: Figure) -> None:
        """Add a matplotlib Figure as a new tab.

        If *name* contains ``/``, the prefix is treated as a subdirectory
        during export and the tab label shows only the basename.
        """
        display_name = name.rsplit("/", 1)[-1] if "/" in name else name
        canvas = FigureCanvasQTAgg(fig)
        self.tabs.addTab(canvas, display_name)
        self._figures[name] = fig
        self.btn_export_all.setEnabled(True)

    def _export_all(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not folder:
            return
        out = Path(folder)
        for name, fig in self._figures.items():
            safe = name.replace(" ", "_")
            save_path = out / f"{safe}.pdf"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
