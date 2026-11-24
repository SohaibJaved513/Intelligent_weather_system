from __future__ import annotations

import numpy as np
import pandas as pd


def add_cyclical_features(data: pd.DataFrame) -> pd.DataFrame:

    result = data.copy()
    hour = result.index.hour
    day_of_year = result.index.dayofyear

    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    result["month"] = result.index.month
    result["day_of_week"] = result.index.dayofweek
    return result


def build_features(
    data: pd.DataFrame,
    target_column: str,
    sensor_columns: tuple[str, ...],
) -> pd.DataFrame:


    result = pd.DataFrame(index=data.index)
    target = data[target_column]
    previous_target = target.shift(1)

                             
    result["temp_lag_1h"] = target.shift(1)
    result["temp_lag_24h"] = target.shift(24)
    result["temp_lag_7d"] = target.shift(24 * 7)

                                                      
    result["temp_rolling_mean_24h"] = previous_target.rolling(24).mean()
    result["temp_rolling_mean_7d"] = previous_target.rolling(24 * 7).mean()
    result["temp_rolling_std_24h"] = previous_target.rolling(24).std()
    result["temp_rolling_std_7d"] = previous_target.rolling(24 * 7).std()

                                              
    result["temp_ewm_24h"] = previous_target.ewm(span=24, adjust=False).mean()
    result["temp_ewm_7d"] = previous_target.ewm(
        span=24 * 7, adjust=False
    ).mean()

                                                                             
    safe_names = {
        "Humidity": "humidity",
        "Pressure (millibars)": "pressure",
        "Wind Speed (km/h)": "wind_speed",
        "Visibility (km)": "visibility",
    }
    for source_column in sensor_columns:
        prefix = safe_names.get(
            source_column,
            source_column.lower().replace(" ", "_").replace("(", "").replace(")", ""),
        )
        result[f"{prefix}_lag_1h"] = data[source_column].shift(1)
        result[f"{prefix}_lag_24h"] = data[source_column].shift(24)

    result = add_cyclical_features(result)
    result["target"] = target
    return result.dropna()


def split_features_and_target(
    model_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    return model_data.drop(columns="target"), model_data["target"]


def build_next_hour_features(
    data: pd.DataFrame,
    target_column: str,
    sensor_columns: tuple[str, ...],
) -> pd.DataFrame:

    future_time = data.index.max() + pd.Timedelta(hours=1)
    target = data[target_column]

    row: dict[str, float | int] = {
        "temp_lag_1h": float(target.iloc[-1]),
        "temp_lag_24h": float(target.loc[future_time - pd.Timedelta(hours=24)]),
        "temp_lag_7d": float(target.loc[future_time - pd.Timedelta(days=7)]),
        "temp_rolling_mean_24h": float(target.tail(24).mean()),
        "temp_rolling_mean_7d": float(target.tail(24 * 7).mean()),
        "temp_rolling_std_24h": float(target.tail(24).std()),
        "temp_rolling_std_7d": float(target.tail(24 * 7).std()),
        "temp_ewm_24h": float(target.ewm(span=24, adjust=False).mean().iloc[-1]),
        "temp_ewm_7d": float(
            target.ewm(span=24 * 7, adjust=False).mean().iloc[-1]
        ),
    }

    safe_names = {
        "Humidity": "humidity",
        "Pressure (millibars)": "pressure",
        "Wind Speed (km/h)": "wind_speed",
        "Visibility (km)": "visibility",
    }
    for source_column in sensor_columns:
        prefix = safe_names.get(
            source_column,
            source_column.lower().replace(" ", "_").replace("(", "").replace(")", ""),
        )
        row[f"{prefix}_lag_1h"] = float(data[source_column].iloc[-1])
        row[f"{prefix}_lag_24h"] = float(
            data.loc[future_time - pd.Timedelta(hours=24), source_column]
        )

    row["hour_sin"] = float(np.sin(2 * np.pi * future_time.hour / 24))
    row["hour_cos"] = float(np.cos(2 * np.pi * future_time.hour / 24))
    row["day_of_year_sin"] = float(
        np.sin(2 * np.pi * future_time.dayofyear / 365.25)
    )
    row["day_of_year_cos"] = float(
        np.cos(2 * np.pi * future_time.dayofyear / 365.25)
    )
    row["month"] = int(future_time.month)
    row["day_of_week"] = int(future_time.dayofweek)

    return pd.DataFrame([row], index=pd.DatetimeIndex([future_time], name="datetime"))
