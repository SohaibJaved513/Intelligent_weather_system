from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error_safe(
    actual: np.ndarray,
    predicted: np.ndarray,
    epsilon: float = 1e-8,
) -> float:

    valid = np.abs(actual) > epsilon
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs((actual[valid] - predicted[valid]) / actual[valid])) * 100)


def symmetric_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator > 1e-8
    return float(
        np.mean(2 * np.abs(predicted[valid] - actual[valid]) / denominator[valid]) * 100
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE_percent": mean_absolute_percentage_error_safe(actual, predicted),
        "sMAPE_percent": symmetric_mape(actual, predicted),
        "R2": float(r2_score(actual, predicted)),
    }
