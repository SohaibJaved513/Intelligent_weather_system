from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.evaluation import regression_metrics


@dataclass
class ModelSelectionResult:
    best_parameters: dict[str, float | int]
    cv_results: pd.DataFrame


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(X) * (1 - validation_fraction))
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("validation_fraction produces an invalid split")

    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


def create_model(parameters: dict[str, float | int], random_state: int) -> XGBRegressor:
    return XGBRegressor(
        **parameters,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=2,
    )


def select_model_with_time_series_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> ModelSelectionResult:

    candidates = [
        {
            "n_estimators": 120,
            "max_depth": 4,
            "learning_rate": 0.07,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        {
            "n_estimators": 180,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
        },
        {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },
    ]

    splitter = TimeSeriesSplit(n_splits=3)
    rows: list[dict[str, float | int]] = []

    for candidate_id, parameters in enumerate(candidates, start=1):
        fold_rmse: list[float] = []
        for fold_id, (train_index, valid_index) in enumerate(
            splitter.split(X_train), start=1
        ):
            model = create_model(parameters, random_state)
            model.fit(
                X_train.iloc[train_index],
                y_train.iloc[train_index],
            )
            predicted = model.predict(X_train.iloc[valid_index])
            metrics = regression_metrics(
                y_train.iloc[valid_index].to_numpy(), predicted
            )
            fold_rmse.append(metrics["RMSE"])
            rows.append(
                {
                    "candidate": candidate_id,
                    "fold": fold_id,
                    "train_start": str(X_train.index[train_index[0]]),
                    "train_end": str(X_train.index[train_index[-1]]),
                    "validation_start": str(X_train.index[valid_index[0]]),
                    "validation_end": str(X_train.index[valid_index[-1]]),
                    "RMSE": metrics["RMSE"],
                    **parameters,
                }
            )

        rows.append(
            {
                "candidate": candidate_id,
                "fold": 0,
                "train_start": "mean",
                "train_end": "mean",
                "validation_start": "mean",
                "validation_end": "mean",
                "RMSE": float(np.mean(fold_rmse)),
                **parameters,
            }
        )

    results = pd.DataFrame(rows)
    means = results[results["fold"] == 0].sort_values("RMSE")
    best_candidate = int(means.iloc[0]["candidate"])
    best_parameters = candidates[best_candidate - 1]
    return ModelSelectionResult(best_parameters=best_parameters, cv_results=results)


def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict[str, float | int],
    random_state: int,
) -> XGBRegressor:
    model = create_model(parameters, random_state)
    model.fit(X_train, y_train)
    return model


def persistence_baseline(X_valid: pd.DataFrame) -> np.ndarray:

    return X_valid["temp_lag_1h"].to_numpy()
