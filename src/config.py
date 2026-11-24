from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:


    data_path: Path = Path("data/raw/weatherHistory.csv")
    processed_path: Path = Path("data/processed/cleaned_weather.csv")
    output_dir: Path = Path("outputs")

    datetime_column: str = "Formatted Date"
    target_column: str = "Temperature (C)"
    validation_fraction: float = 0.20
    random_state: int = 42

                                                                  
    sensor_columns: tuple[str, ...] = field(
        default=(
            "Humidity",
            "Pressure (millibars)",
            "Wind Speed (km/h)",
            "Visibility (km)",
        )
    )
