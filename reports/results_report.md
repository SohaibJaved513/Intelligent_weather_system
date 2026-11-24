# Weather Prediction Results Report

## 1. Objective

The project forecasts hourly **temperature in degrees Celsius** from the uploaded
`weatherHistory.csv` dataset. Temperature was selected because the file contains
continuous temperature observations, while `Precip Type` is only a categorical
rain/snow label rather than a rainfall-amount target.

## 2. Data preparation

- Original rows: **96,453**
- Duplicate timestamps consolidated: **24**
- Missing hourly timestamps inserted: **3**
- Missing target values before interpolation: **3**
- Missing target values after interpolation: **0**
- Final hourly period: **2005-12-31 23:00:00+00:00** to **2016-12-31 22:00:00+00:00**

Duplicate timestamps were averaged. The three absent hourly records were inserted
with a complete UTC hourly index and filled using time-based interpolation. This
uses neighboring observations and preserves local temporal movement better than a
global mean replacement.

Evidence: `outputs/missing_data_before_after.png`.

## 3. Feature engineering

The custom Pandas/NumPy pipeline creates:

- 1-hour, 24-hour, and 7-day temperature lags;
- 24-hour and 7-day rolling means;
- 24-hour and 7-day rolling standard deviations;
- 24-hour and 7-day exponentially weighted means;
- lagged humidity, pressure, wind speed, and visibility;
- sine/cosine hour-of-day encoding;
- sine/cosine day-of-year encoding;
- month and weekday.

All rolling and exponential features begin with `shift(1)`. Same-hour sensor values
are never used, so each row contains only information available before its target.

## 4. Leakage prevention and validation

The final **20%** of observations is held out as validation data:

- Training: **2006-01-07 23:00:00+00:00** to **2014-10-21 17:00:00+00:00**
- Validation: **2014-10-21 18:00:00+00:00** to **2016-12-31 22:00:00+00:00**

No random shuffle is used. A shuffled split would allow later weather patterns to
enter training while earlier rows appear in validation, producing optimistic and
operationally unrealistic scores.

Model selection uses a three-fold expanding-window `TimeSeriesSplit` only within
the training period. The selected configuration had mean CV RMSE of
**1.0798 °C**.

## 5. Model choice

A lagged-feature XGBoost regressor was selected because the dataset is large,
nonlinear, seasonal, and includes interactions among recent temperature, daily
history, weekly history, and other lagged sensors. The lag and cyclic variables
provide temporal context, while tree boosting handles nonlinear relationships
without requiring a rigid linear seasonal form.

Selected parameters:

```text
{
  "n_estimators": 100,
  "max_depth": 5,
  "learning_rate": 0.08,
  "subsample": 0.8,
  "colsample_bytree": 0.8
}
```

## 6. Held-out validation results

| Metric | XGBoost | Previous-hour baseline |
|---|---:|---:|
| RMSE (°C) | 0.8515 | 1.2100 |
| MAE (°C) | 0.6372 | 0.8412 |
| MAPE (%) | 31.9252 | 31.7194 |
| sMAPE (%) | 14.3969 | 15.6503 |
| R² | 0.9912 | 0.9823 |

MAPE is reported because the assignment requires it, but it becomes unstable when
actual temperature is near 0°C. RMSE and MAE are therefore more interpretable for
this target.

Prediction chart: `outputs/actual_vs_predicted.png`.

## 7. Interpretability

The project includes two interpretability views:

1. XGBoost gain importance in `outputs/feature_importance_gain.png`.
2. Native XGBoost SHAP contributions in `outputs/shap_global_importance.png`.

The strongest drivers are:

- **temp_lag_1h**: mean absolute SHAP 5.4576
- **temp_lag_24h**: mean absolute SHAP 1.2124
- **temp_ewm_24h**: mean absolute SHAP 0.7836
- **hour_cos**: mean absolute SHAP 0.5496
- **hour_sin**: mean absolute SHAP 0.4003

These results show that recent temperature history and daily recurrence dominate
the one-hour forecast. The local explanation file,
`outputs/shap_local_explanation.csv`, gives signed feature contributions for one
specific recent validation prediction.

## 8. Next-hour forecast

After fitting the final model, the pipeline creates a one-step forecast for the
hour immediately following the last dataset timestamp. The reproducible result is
stored in `outputs/next_hour_forecast.json`, and the trained booster is saved as
`outputs/weather_xgb_model.json`.

## 9. Limitations

- This is a one-step-ahead hourly model evaluated on historical observations.
- MAPE is sensitive to temperatures near zero.
- The dataset does not include numeric rainfall accumulation, so rainfall amount
  forecasting is outside the available target columns.
- Operational forecasting would require a defined retraining schedule and live
  sensor ingestion.
