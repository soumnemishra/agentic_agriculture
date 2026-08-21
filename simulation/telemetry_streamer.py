"""
ACA Simulation Layer — IoT Telemetry Streamer
==============================================

Provides a deterministic, synchronized sensor data streamer that merges
and yields multi-modal IoT microclimate telemetry row-by-row for real-time
agentic perception and digital-twin simulations.

Synchronized Datasets (8 channels, 100 rows, 18-sec intervals):
    1. Environment Humidity (%)
    2. Environment Light Intensity (Lux)
    3. Environment Temperature (°C)
    4. Soil Moisture (%)
    5. Soil pH
    6. Soil Temperature (°C)
    7. Solar Panel Battery Voltage (V)
    8. Water TDS (mg/L)

Design Decisions:
    - CPU-only operation using pandas / openpyxl / zipfile parsing.
    - Deterministic merge on ``Entry_id`` / timestamp.
    - Yields clean, type-hinted dictionaries conforming to the ACA schema.
    - Supports generator stepping, iteration, looping, and state inspection.
"""

from __future__ import annotations

import os
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from aca.logging_config import get_logger

logger = get_logger("simulation.telemetry_streamer")

# Default directory path for IoT datasets
DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "Smart Agriculture and Plant Health Monitoring using IoT",
)

# Standard channel metadata mapping: filename -> (output_field_name, unit, expected_header_keyword)
CHANNEL_MAPPINGS: Dict[str, Dict[str, str]] = {
    "Environment Humidity.xlsx": {
        "field": "environment_humidity",
        "unit": "%",
        "header_prefix": "Environment Humidity",
    },
    "Environment Light Intensity.xlsx": {
        "field": "environment_light_lux",
        "unit": "Lux",
        "header_prefix": "Environment Light Intensity",
    },
    "Environment Temperature.xlsx": {
        "field": "environment_temperature_c",
        "unit": "°C",
        "header_prefix": "Environment Temperature",
    },
    "Soil Moisture.xlsx": {
        "field": "soil_moisture",
        "unit": "%",
        "header_prefix": "Soil Moisture",
    },
    "Soil pH.xlsx": {
        "field": "soil_ph",
        "unit": "pH",
        "header_prefix": "Soil pH",
    },
    "Soil Temperature.xlsx": {
        "field": "soil_temperature_c",
        "unit": "°C",
        "header_prefix": "Soil Temperature",
    },
    "Solar Panel Battery Voltage.xlsx": {
        "field": "solar_battery_voltage",
        "unit": "V",
        "header_prefix": "Solar Panel Battery Voltage",
    },
    "Water TDS.xlsx": {
        "field": "water_tds",
        "unit": "mg/L",
        "header_prefix": "Water TDS",
    },
}


@dataclass
class TelemetryRecord:
    """
    Strongly typed dataclass representing a single synchronized IoT reading.
    """
    entry_id: int
    timestamp: str
    environment_humidity: float
    environment_light_lux: float
    environment_temperature_c: float
    soil_moisture: float
    soil_ph: float
    soil_temperature_c: float
    solar_battery_voltage: float
    water_tds: float
    raw_units: Dict[str, str] = field(default_factory=lambda: {
        "environment_humidity": "%",
        "environment_light_lux": "Lux",
        "environment_temperature_c": "°C",
        "soil_moisture": "%",
        "soil_ph": "pH",
        "soil_temperature_c": "°C",
        "solar_battery_voltage": "V",
        "water_tds": "mg/L",
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a dictionary payload for ACA Observations."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "environment_humidity": self.environment_humidity,
            "environment_light_lux": self.environment_light_lux,
            "environment_temperature_c": self.environment_temperature_c,
            "soil_moisture": self.soil_moisture,
            "soil_ph": self.soil_ph,
            "soil_temperature_c": self.soil_temperature_c,
            "solar_battery_voltage": self.solar_battery_voltage,
            "water_tds": self.water_tds,
            "units": self.raw_units,
        }


def _read_excel_fallback(filepath: str) -> List[Dict[str, Any]]:
    """
    Pure Python zero-dependency reader for simple openxml .xlsx files.
    Used when openpyxl or xlrd is not installed in the active environment.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with zipfile.ZipFile(filepath, "r") as z:
        # 1. Parse shared strings table if present
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in tree.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                t = si.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                shared_strings.append(t.text if t is not None and t.text is not None else "")

        # 2. Parse sheet1.xml
        sheet_tree = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows: List[List[str]] = []
        for r in sheet_tree.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            row_vals: List[str] = []
            for c in r.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                t_attr = c.get("t")
                v_tag = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                val = v_tag.text if v_tag is not None and v_tag.text is not None else ""
                if t_attr == "s" and val.isdigit() and int(val) < len(shared_strings):
                    val = shared_strings[int(val)]
                row_vals.append(val)
            if any(row_vals):
                rows.append(row_vals)

    # In this dataset, Row 0 is the title, Row 1 is header: [Date & Time Created, Entry_id, Value]
    if len(rows) < 2:
        return []

    data_rows: List[Dict[str, Any]] = []
    for r in rows[2:]:
        if len(r) >= 3 and r[1].isdigit():
            entry_id = int(r[1])
            ts = r[0].strip()
            try:
                val = float(r[2])
            except (ValueError, TypeError):
                val = 0.0
            data_rows.append({"timestamp": ts, "entry_id": entry_id, "value": val})

    return data_rows


class IoTStreamer:
    """
    Deterministic IoT microclimate telemetry streamer.

    Synchronizes 8 Excel files on ``Entry_id`` and streams multi-modal
    environmental observations sequentially on CPU.

    Args:
        dataset_dir: Directory containing the 8 Excel files.
        loop: Whether to loop indefinitely when the dataset ends.
        auto_load: Whether to load and merge data immediately in ``__init__``.

    Example::

        streamer = IoTStreamer()
        for sample in streamer:
            print(sample["entry_id"], sample["environment_temperature_c"])
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        loop: bool = True,
        auto_load: bool = True,
    ) -> None:
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.loop = loop
        self._records: List[TelemetryRecord] = []
        self._current_index: int = 0
        self._total_records: int = 0

        if auto_load:
            self.load()

    def load(self) -> None:
        """Load and merge the 8 Excel files into synchronized records."""
        logger.info("Loading synchronized IoT dataset from %s", self.dataset_dir)
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        channel_data: Dict[str, Dict[int, float]] = {}
        timestamp_map: Dict[int, str] = {}
        all_entry_ids: set[int] = set()

        for filename, meta in CHANNEL_MAPPINGS.items():
            field_name = meta["field"]
            filepath = os.path.join(self.dataset_dir, filename)

            rows: List[Dict[str, Any]] = []
            try:
                import pandas as pd
                try:
                    df = pd.read_excel(filepath, skiprows=1)
                    # Standardize column names
                    date_col = [c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()][0]
                    entry_col = [c for c in df.columns if "entry" in str(c).lower()][0]
                    val_col = [c for c in df.columns if c not in (date_col, entry_col)][0]

                    for _, row in df.iterrows():
                        try:
                            eid = int(row[entry_col])
                            val = float(row[val_col])
                            ts = str(row[date_col]).strip()
                            rows.append({"timestamp": ts, "entry_id": eid, "value": val})
                        except (ValueError, TypeError):
                            continue
                except Exception:
                    rows = _read_excel_fallback(filepath)
            except Exception:
                rows = _read_excel_fallback(filepath)

            field_dict: Dict[int, float] = {}
            for r in rows:
                eid = r["entry_id"]
                field_dict[eid] = r["value"]
                if eid not in timestamp_map:
                    timestamp_map[eid] = r["timestamp"]
                all_entry_ids.add(eid)

            channel_data[field_name] = field_dict
            logger.debug("Loaded %d rows for channel %s", len(field_dict), field_name)

        sorted_eids = sorted(all_entry_ids)
        self._records = []

        for eid in sorted_eids:
            rec = TelemetryRecord(
                entry_id=eid,
                timestamp=timestamp_map.get(eid, ""),
                environment_humidity=channel_data.get("environment_humidity", {}).get(eid, 0.0),
                environment_light_lux=channel_data.get("environment_light_lux", {}).get(eid, 0.0),
                environment_temperature_c=channel_data.get("environment_temperature_c", {}).get(eid, 0.0),
                soil_moisture=channel_data.get("soil_moisture", {}).get(eid, 0.0),
                soil_ph=channel_data.get("soil_ph", {}).get(eid, 7.0),
                soil_temperature_c=channel_data.get("soil_temperature_c", {}).get(eid, 0.0),
                solar_battery_voltage=channel_data.get("solar_battery_voltage", {}).get(eid, 0.0),
                water_tds=channel_data.get("water_tds", {}).get(eid, 0.0),
            )
            self._records.append(rec)

        self._total_records = len(self._records)
        self._current_index = 0
        logger.info("Successfully merged %d synchronized IoT telemetry records", self._total_records)

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Advance one step and return the current synchronized telemetry row.

        Returns:
            Dictionary with all 8 channel values and metadata, or None if
            at end and loop is False.
        """
        if self._total_records == 0:
            return None

        if self._current_index >= self._total_records:
            if self.loop:
                self._current_index = 0
            else:
                return None

        record = self._records[self._current_index]
        self._current_index += 1
        return record.to_dict()

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Peek at the current telemetry row without advancing the pointer."""
        if self._total_records == 0:
            return None
        idx = max(0, min(self._current_index, self._total_records - 1))
        return self._records[idx].to_dict()

    def reset(self, loop: Optional[bool] = None) -> None:
        """Reset the streaming pointer back to the first entry."""
        self._current_index = 0
        if loop is not None:
            self.loop = loop
        logger.debug("IoTStreamer reset to index 0 (loop=%s)", self.loop)

    @property
    def total_records(self) -> int:
        """Total number of synchronized records in dataset."""
        return self._total_records

    @property
    def current_index(self) -> int:
        """Current zero-based index pointer."""
        return self._current_index

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterator protocol."""
        return self

    def __next__(self) -> Dict[str, Any]:
        """Next item in iteration."""
        res = self.step()
        if res is None:
            raise StopIteration
        return res
