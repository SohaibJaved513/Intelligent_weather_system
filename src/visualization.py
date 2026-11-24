from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_missing_data_evidence(
    before: pd.DataFrame,
    after: pd.DataFrame,
    target_column: str,
    missing_timestamps: list[str],
    output_path: Path,
) -> None:
    if missing_timestamps:
        center = pd.Timestamp(missing_timestamps[0])
    else:
        center = after.index[len(after) // 2]

    start = center - pd.Timedelta(hours=36)
    end = center + pd.Timedelta(hours=36)
    before_window = before.loc[start:end, target_column]
    after_window = after.loc[start:end, target_column]

    plt.figure(figsize=(12, 5))
    plt.plot(before_window.index, before_window, marker="o", label="Before interpolation")
    plt.plot(after_window.index, after_window, label="After interpolation")
    if missing_timestamps:
        for timestamp in missing_timestamps:
            value = pd.Timestamp(timestamp)
            if start <= value <= end:
                plt.axvline(value, linestyle="--", alpha=0.6)
    plt.title("Missing Hour Before and After Time Interpolation")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def plot_actual_vs_predicted(results: pd.DataFrame, output_path: Path) -> None:
    window = results.tail(24 * 30)
    plt.figure(figsize=(13, 5))
    plt.plot(window["datetime"], window["actual"], label="Actual")
    plt.plot(window["datetime"], window["predicted"], label="Predicted")
    plt.title("Actual vs Predicted Temperature — Final 30 Validation Days")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def plot_importance(
    importance: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
) -> None:
    top = importance.head(15).sort_values(value_column)
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top[value_column])
    plt.title(title)
    plt.xlabel(value_column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def plot_residuals(results: pd.DataFrame, output_path: Path) -> None:
    residuals = results["actual"] - results["predicted"]
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50)
    plt.title("Validation Residual Distribution")
    plt.xlabel("Actual − Predicted Temperature (°C)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()
