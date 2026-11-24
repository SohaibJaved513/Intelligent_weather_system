# Intelligent Weather Prediction Pipeline

A complete, modular time-series project built from the supplied
`weatherHistory.csv` dataset.

## What the project predicts

The pipeline predicts the next hourly **Temperature (C)** value. The dataset has a
categorical `Precip Type` field but no numeric rainfall-amount column, so temperature
is the cleanest continuous forecasting target.

## Requirement coverage

- Genuine time-aware XGBoost model with lagged variables
- 24-hour and 7-day lags
- Rolling averages and rolling standard deviations
- Exponential smoothing features
- Hour and day-of-year sine/cosine encoding
- Complete hourly reindexing and time interpolation
- Strict chronological 80/20 split
- Three-fold expanding-window `TimeSeriesSplit`
- RMSE, MAPE, MAE, sMAPE, and R²
- Previous-hour persistence baseline
- Predicted-vs-actual visualization
- Gain importance and native XGBoost SHAP contributions
- Missing-data before/after evidence
- Saved trained model and reproducible next-hour forecast
- Modular source files, report, tests, and exploration notebook

## Project structure

```text
weather_prediction_submission/
├── data/
│   ├── raw/weatherHistory.csv
│   └── processed/cleaned_weather.csv
├── notebooks/01_exploration.ipynb
├── outputs/
├── reports/results_report.md
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── interpretability.py
│   ├── modeling.py
│   └── visualization.py
├── tests/
├── README.md
├── STEP_BY_STEP.md
├── requirements.txt
└── run_pipeline.py
```

## Run from the project folder

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
# or: python main.py
```

Use a different dataset path without editing code:

```bash
python run_pipeline.py --data-path path/to/weatherHistory.csv
```

## Main submission files

- `reports/results_report.md`
- `outputs/metrics_and_summary.json`
- `outputs/validation_predictions.csv`
- `outputs/actual_vs_predicted.png`
- `outputs/missing_data_before_after.png`
- `outputs/feature_importance_gain.png`
- `outputs/shap_global_importance.png`
- `outputs/shap_local_explanation.csv`
- `outputs/time_series_cv_results.csv`
- `outputs/weather_xgb_model.json`
- `outputs/next_hour_forecast.json`

## Why the split is not random

Weather observations are sequential. Random splitting mixes future periods into
training and can make the validation score unrealistically strong. This pipeline
uses older observations for training and the newest 20% for validation.

## Code explanation

All code comments and explanatory docstrings have been removed from the Python files. The complete module-by-module explanation is available in [`CODE_EXPLANATION.md`](CODE_EXPLANATION.md).
