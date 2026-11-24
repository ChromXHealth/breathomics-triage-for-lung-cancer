"""Input and output helpers for analysis workflows."""

from pathlib import Path

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a table without a DataFrame index, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
