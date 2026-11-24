import numpy as np
import pandas as pd

from src.feature_engineering import build_features


def test_required_feature_columns_exist() -> None:
    index = pd.date_range("2020-01-01", periods=300, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "Temperature (C)": np.arange(300, dtype=float),
            "Humidity": 0.5,
            "Pressure (millibars)": 1000.0,
            "Wind Speed (km/h)": 10.0,
            "Visibility (km)": 9.0,
        },
        index=index,
    )
    features = build_features(
        data,
        "Temperature (C)",
        (
            "Humidity",
            "Pressure (millibars)",
            "Wind Speed (km/h)",
            "Visibility (km)",
        ),
    )
    expected = {
        "temp_lag_24h",
        "temp_lag_7d",
        "temp_rolling_mean_24h",
        "temp_rolling_std_7d",
        "temp_ewm_24h",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    }
    assert expected.issubset(features.columns)


def test_rolling_mean_uses_only_previous_values() -> None:
    index = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    values = np.arange(200, dtype=float)
    data = pd.DataFrame(
        {
            "Temperature (C)": values,
            "Humidity": 0.5,
            "Pressure (millibars)": 1000.0,
            "Wind Speed (km/h)": 10.0,
            "Visibility (km)": 9.0,
        },
        index=index,
    )
    features = build_features(
        data,
        "Temperature (C)",
        (
            "Humidity",
            "Pressure (millibars)",
            "Wind Speed (km/h)",
            "Visibility (km)",
        ),
    )
    timestamp = index[168]
    expected = values[144:168].mean()
    assert features.loc[timestamp, "temp_rolling_mean_24h"] == expected
