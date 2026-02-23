"""Load experiment metadata from Excel files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metadata(
    filepath: str | Path,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    """Load an experiment metadata Excel file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``info_mice.xlsx`` file.
    sheet_name : str
        Sheet name to read.

    Returns
    -------
    pd.DataFrame
        Metadata table.  Column names are normalised to the codebase
        convention (``mouse`` instead of ``animal_id``, ``sex`` instead
        of ``gender``).
    """
    df = pd.read_excel(
        Path(filepath), engine="openpyxl", sheet_name=sheet_name
    )

    # Normalize column names to the convention used throughout the codebase
    rename_map: dict[str, str] = {}
    cols = df.columns.tolist()
    if "animal_id" in cols and "mouse" not in cols:
        rename_map["animal_id"] = "mouse"
    if "gender" in cols and "sex" not in cols:
        rename_map["gender"] = "sex"
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def get_animals_by_group(
    metadata: pd.DataFrame,
    group_col: str = "group_name",
) -> dict[str, list]:
    """Return a dict mapping group names to lists of animal IDs.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table.
    group_col : str
        Column name containing group labels.

    Returns
    -------
    dict
        ``{group_name: [animal_id, ...], ...}``
    """
    groups = metadata[group_col].unique().tolist()
    return {
        gr: metadata[metadata[group_col] == gr]["mouse"].tolist()
        for gr in groups
    }
