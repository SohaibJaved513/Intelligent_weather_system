from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.data_processing import clean_weather_data, load_weather_data, save_processed_data
from src.evaluation import regression_metrics
from src.feature_engineering import (
    build_features,
    build_next_hour_features,
    split_features_and_target,
)
from src.interpretability import gain_importance, shap_contribution_summary
from src.modeling import (
    fit_final_model,
    persistence_baseline,
    select_model_with_time_series_cv,
    time_based_split,
)
from src.visualization import (
    plot_actual_vs_predicted,
    plot_importance,
    plot_missing_data_evidence,
    plot_residuals,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather forecasting pipeline")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/weatherHistory.csv"),
        help="Path to weatherHistory.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = ProjectConfig(data_path=args.data_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_weather_data(config.data_path)
    cleaned, before_interpolation, cleaning = clean_weather_data(
        raw,
        config.datetime_column,
        config.target_column,
        config.sensor_columns,
    )
    save_processed_data(cleaned, config.processed_path)

    model_data = build_features(
        cleaned,
        config.target_column,
        config.sensor_columns,
    )
    X, y = split_features_and_target(model_data)
    X_train, X_valid, y_train, y_valid = time_based_split(
        X, y, config.validation_fraction
    )

    selection = select_model_with_time_series_cv(
        X_train, y_train, config.random_state
    )
    selection.cv_results.to_csv(config.output_dir / "time_series_cv_results.csv", index=False)

    model = fit_final_model(
        X_train,
        y_train,
        selection.best_parameters,
        config.random_state,
    )
    predicted = model.predict(X_valid)
    model.save_model(config.output_dir / "weather_xgb_model.json")

    next_hour_features = build_next_hour_features(
        cleaned, config.target_column, config.sensor_columns
    ).reindex(columns=X.columns)
    next_hour_prediction = float(model.predict(next_hour_features)[0])
    next_hour_forecast = {
        "forecast_timestamp": str(next_hour_features.index[0]),
        "predicted_temperature_C": next_hour_prediction,
        "forecast_type": "one-step-ahead hourly forecast",
    }
    with open(
        config.output_dir / "next_hour_forecast.json", "w", encoding="utf-8"
    ) as file:
        json.dump(next_hour_forecast, file, indent=2)

    baseline_predicted = persistence_baseline(X_valid)

    model_metrics = regression_metrics(y_valid.to_numpy(), predicted)
    baseline_metrics = regression_metrics(y_valid.to_numpy(), baseline_predicted)

    predictions = pd.DataFrame(
        {
            "datetime": y_valid.index,
            "actual": y_valid.to_numpy(),
            "predicted": predicted,
            "baseline_previous_hour": baseline_predicted,
        }
    )
    predictions["absolute_error"] = (
        predictions["actual"] - predictions["predicted"]
    ).abs()
    predictions.to_csv(config.output_dir / "validation_predictions.csv", index=False)

    gain = gain_importance(model, list(X.columns))
    gain.to_csv(config.output_dir / "feature_importance_gain.csv", index=False)

    shap_global, shap_local = shap_contribution_summary(model, X_valid)
    shap_global.to_csv(config.output_dir / "shap_global_importance.csv", index=False)
    shap_local.to_csv(config.output_dir / "shap_local_explanation.csv", index=False)

    plot_missing_data_evidence(
        before_interpolation,
        cleaned,
        config.target_column,
        cleaning.missing_timestamps,
        config.output_dir / "missing_data_before_after.png",
    )
    plot_actual_vs_predicted(
        predictions,
        config.output_dir / "actual_vs_predicted.png",
    )
    plot_importance(
        gain,
        "normalized_gain",
        "XGBoost Gain-Based Feature Importance",
        config.output_dir / "feature_importance_gain.png",
    )
    plot_importance(
        shap_global,
        "mean_absolute_shap",
        "Global SHAP Contribution Importance",
        config.output_dir / "shap_global_importance.png",
    )
    plot_residuals(predictions, config.output_dir / "validation_residuals.png")

    best_cv_row = (
        selection.cv_results[selection.cv_results["fold"] == 0]
        .sort_values("RMSE")
        .iloc[0]
    )
    summary = {
        "dataset": {
            **cleaning.__dict__,
            "rows_after_feature_engineering": int(len(model_data)),
        },
        "split": {
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
            "training_start": str(X_train.index.min()),
            "training_end": str(X_train.index.max()),
            "validation_start": str(X_valid.index.min()),
            "validation_end": str(X_valid.index.max()),
        },
        "model": {
            "name": "XGBRegressor",
            "best_parameters": selection.best_parameters,
            "best_mean_cv_RMSE": float(best_cv_row["RMSE"]),
        },
        "next_hour_forecast": next_hour_forecast,
        "validation_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "top_gain_features": gain.head(10).to_dict(orient="records"),
        "top_shap_features": shap_global.head(10).to_dict(orient="records"),
    }
    with open(config.output_dir / "metrics_and_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


    print(json.dumps(summary["validation_metrics"], indent=2))


if __name__ == "__main__":
    main()
