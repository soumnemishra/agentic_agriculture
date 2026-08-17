"""
ACA IoT Telemetry Streamer
==========================

Simulates real-time IoT sensor telemetry streaming for the Agricultural
Cognitive Architecture (ACA) by parsing, merging, and iterating over 
synchronized multi-sensor Excel files on CPU using Pandas.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterator, List, Optional
import pandas as pd

from aca.logging_config import get_logger

logger = get_logger("simulation.telemetry_streamer")


class IoTStreamer:
    """
    Simulates a streaming IoT telemetry node by reading 8 synchronized Excel datasets.

    Merges sensor channels on ``Entry_id`` to ensure zero-phase temporal alignment
    across all environmental, soil, power, and water parameters.

    Args:
        dataset_dir: Path to folder containing the 8 Excel files.
        loop: Whether to loop back to start when reaching end of stream.
    """

    def __init__(
        self,
        dataset_dir: str = r"d:\agentic_agriculture\datasets\Smart Agriculture and Plant Health Monitoring using IoT",
        loop: bool = True,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.loop = loop
        self._current_index = 0
        self._df: pd.DataFrame = pd.DataFrame()
        self._load_and_merge_datasets()

    def _load_and_merge_datasets(self) -> None:
        """Loads all .xlsx files in dataset_dir and performs inner join on Entry_id."""
        pattern = os.path.join(self.dataset_dir, "*.xlsx")
        files = sorted(glob.glob(pattern))

        if not files:
            logger.error("No Excel telemetry files found in %s", self.dataset_dir)
            raise FileNotFoundError(f"No Excel telemetry files found in {self.dataset_dir}")

        dfs: List[pd.DataFrame] = []
        for file_path in files:
            fname = os.path.basename(file_path).replace(".xlsx", "")
            # Header is on row 1 (0-indexed)
            df = pd.read_excel(file_path, header=1).dropna(how="all", axis=1)

            # Find measurement column and timestamp column
            val_col = [c for c in df.columns if "Measurement" in c or "Mesaurement" in c][0]
            date_col = [c for c in df.columns if "Date" in c or "Time" in c][0]
            entry_col = [c for c in df.columns if "Entry" in c][0]

            df_clean = df[[entry_col, date_col, val_col]].copy()
            df_clean = df_clean.rename(columns={
                entry_col: "Entry_id",
                date_col: "Timestamp",
                val_col: fname
            })
            dfs.append(df_clean)

        # Synchronize and merge on Entry_id
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(
                merged,
                df.drop(columns=["Timestamp"], errors="ignore"),
                on="Entry_id",
                how="outer"
            )

        merged = merged.sort_values(by="Entry_id").reset_index(drop=True)
        self._df = merged
        logger.info(
            "IoTStreamer loaded %d synchronized telemetry rows across %d channels from %s",
            len(self._df),
            len(self._df.columns) - 2,
            self.dataset_dir,
        )

    @property
    def total_records(self) -> int:
        """Total number of records in the dataset."""
        return len(self._df)

    @property
    def current_index(self) -> int:
        """Current row pointer in the stream."""
        return self._current_index

    def reset(self) -> None:
        """Reset stream pointer back to beginning."""
        self._current_index = 0
        logger.debug("IoTStreamer reset to index 0")

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Advance stream by one tick and return the synchronized row.

        Returns:
            Dictionary containing sensor telemetry data for the current tick,
            or ``None`` if end of stream reached and loop=False.
        """
        if self._df.empty:
            return None

        if self._current_index >= len(self._df):
            if self.loop:
                self._current_index = 0
                logger.debug("IoTStreamer reached end of stream, looping back to start.")
            else:
                logger.info("IoTStreamer reached end of stream.")
                return None

        row = self._df.iloc[self._current_index].to_dict()
        self._current_index += 1
        return row

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterable interface yielding rows sequentially."""
        return self

    def __next__(self) -> Dict[str, Any]:
        """Next item generator interface."""
        res = self.step()
        if res is None:
            raise StopIteration
        return res
