from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor


def gain_importance(model: XGBRegressor, feature_names: list[str]) -> pd.DataFrame:
    raw = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": [float(raw.get(name, 0.0)) for name in feature_names],
        }
    )
    total = importance["gain"].sum()
    importance["normalized_gain"] = (
        importance["gain"] / total if total > 0 else 0.0
    )
    return importance.sort_values("normalized_gain", ascending=False)


def shap_contribution_summary(
    model: XGBRegressor,
    X_valid: pd.DataFrame,
    sample_size: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:


    if len(X_valid) > sample_size:
        sample_positions = np.linspace(
            0, len(X_valid) - 1, sample_size, dtype=int
        )
        sample = X_valid.iloc[sample_positions]
    else:
        sample = X_valid

    matrix = xgb.DMatrix(sample, feature_names=list(sample.columns))
    contributions = model.get_booster().predict(matrix, pred_contribs=True)

    feature_contributions = contributions[:, :-1]
    global_summary = pd.DataFrame(
        {
            "feature": sample.columns,
            "mean_absolute_shap": np.abs(feature_contributions).mean(axis=0),
            "mean_shap": feature_contributions.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)

                                                       
    local_position = -1
    local = pd.DataFrame(
        {
            "feature": sample.columns,
            "feature_value": sample.iloc[local_position].to_numpy(),
            "shap_contribution": feature_contributions[local_position],
        }
    )
    local["absolute_contribution"] = local["shap_contribution"].abs()
    local = local.sort_values("absolute_contribution", ascending=False)
    local["timestamp"] = str(sample.index[local_position])
    local["base_value"] = float(contributions[local_position, -1])
    return global_summary, local
