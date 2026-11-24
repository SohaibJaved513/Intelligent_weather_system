from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CleaningSummary:
    original_rows: int
    invalid_timestamps: int
    duplicate_timestamps: int
    missing_hours_added: int
    target_missing_before: int
    target_missing_after: int
    start_time: str
    end_time: str
    missing_timestamps: list[str]


def load_weather_data(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_weather_data(
    raw: pd.DataFrame,
    datetime_column: str,
    target_column: str,
    sensor_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningSummary]:


    required = [datetime_column, target_column, *sensor_columns]
    missing_columns = [column for column in required if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = raw[required].copy()
    data[datetime_column] = pd.to_datetime(
        data[datetime_column], utc=True, errors="coerce"
    )

    invalid_timestamps = int(data[datetime_column].isna().sum())
    data = data.dropna(subset=[datetime_column])

    for column in [target_column, *sensor_columns]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values(datetime_column)
    duplicate_timestamps = int(data[datetime_column].duplicated().sum())

                                                                           
    data = data.groupby(datetime_column, as_index=True).mean(numeric_only=True)
    data.index.name = "datetime"

    complete_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="h",
        tz="UTC",
    )
    missing_timestamps = complete_index.difference(data.index)

    before_interpolation = data.reindex(complete_index)
    before_interpolation.index.name = "datetime"
    target_missing_before = int(before_interpolation[target_column].isna().sum())

    cleaned = before_interpolation.copy()
    numeric_columns = [target_column, *sensor_columns]
    cleaned[numeric_columns] = cleaned[numeric_columns].interpolate(method="time")

                                                                      
    cleaned[numeric_columns] = cleaned[numeric_columns].ffill().bfill()
    target_missing_after = int(cleaned[target_column].isna().sum())

    summary = CleaningSummary(
        original_rows=int(len(raw)),
        invalid_timestamps=invalid_timestamps,
        duplicate_timestamps=duplicate_timestamps,
        missing_hours_added=int(len(missing_timestamps)),
        target_missing_before=target_missing_before,
        target_missing_after=target_missing_after,
        start_time=str(cleaned.index.min()),
        end_time=str(cleaned.index.max()),
        missing_timestamps=[str(value) for value in missing_timestamps],
    )
    return cleaned, before_interpolation, summary


def save_processed_data(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path)
