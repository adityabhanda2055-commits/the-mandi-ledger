"""Crop Price Prediction Model - reproducible monthly baseline.

The included demo CSV is synthetic. It is not actual mandi/AGMARKNET data.
Expected input columns:
    date, modal_price_rs_per_quintal
One row per consecutive month for ONE crop + ONE market + ONE unit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "modal_price_rs_per_quintal"
FEATURES = ["month", "year", "lag_1", "lag_2", "lag_12", "rolling_mean_3"]


def load_monthly_data(path_or_buffer) -> pd.DataFrame:
    """Load and validate a consecutive monthly price series."""
    df = pd.read_csv(path_or_buffer)
    required = {"date", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV must contain columns 'date' and 'modal_price_rs_per_quintal'. "
            f"Missing: {', '.join(sorted(missing))}"
        )

    # If the source export retains identifiers, reject a mixed series instead of
    # silently averaging different crops, markets, districts or varieties together.
    identifier_aliases = {
        "state", "statename", "district", "districtname", "market", "marketname",
        "commodity", "commodityname", "variety", "crop", "cropname"
    }
    for column in df.columns:
        normalized = "".join(ch for ch in str(column).lower() if ch.isalnum())
        if normalized in identifier_aliases and df[column].dropna().nunique() > 1:
            raise ValueError(
                f"Column '{column}' contains multiple values. Filter the CSV to one "
                "crop, one market, one variety and one consistent location before training."
            )

    df = df[["date", TARGET]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["date", TARGET]).sort_values("date")
    if df.empty:
        raise ValueError("No valid date/price rows were found in the CSV.")
    if (df[TARGET] <= 0).any():
        raise ValueError("Prices must be positive. Correct or remove non-positive values first.")

    # Normalize dates to calendar month starts. If there are multiple daily records
    # within a month, average the daily modal prices to form one monthly observation.
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.groupby("date", as_index=False)[TARGET].mean().sort_values("date")

    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    actual = pd.DatetimeIndex(df["date"])
    missing_months = expected.difference(actual)
    if len(missing_months):
        sample = ", ".join(d.strftime("%Y-%m") for d in missing_months[:6])
        raise ValueError(
            "The monthly series has missing months (for example: " + sample + "). "
            "Choose a continuous date window or document a defensible missing-data method. "
            "The script will not silently invent prices."
        )
    if len(df) < 24:
        raise ValueError("At least 24 consecutive monthly observations are required for lag_12 features.")
    return df.reset_index(drop=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year
    out["lag_1"] = out[TARGET].shift(1)
    out["lag_2"] = out[TARGET].shift(2)
    out["lag_12"] = out[TARGET].shift(12)
    out["rolling_mean_3"] = out[TARGET].shift(1).rolling(3).mean()
    return out.dropna().reset_index(drop=True)


def score(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "MAE_Rs_per_quintal": float(mean_absolute_error(actual, predicted)),
        "RMSE_Rs_per_quintal": float(math.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def next_month_features(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.DataFrame]:
    next_date = df["date"].max() + pd.offsets.MonthBegin(1)
    prices = df[TARGET].to_numpy(dtype=float)
    row = {
        "month": next_date.month,
        "year": next_date.year,
        "lag_1": prices[-1],
        "lag_2": prices[-2],
        "lag_12": prices[-12],
        "rolling_mean_3": float(np.mean(prices[-3:])),
    }
    return next_date, pd.DataFrame([row], columns=FEATURES)


def train_evaluate(df: pd.DataFrame, output_dir: Path, data_status: str = "PROVENANCE NOT VERIFIED") -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    featured = make_features(df)
    if len(featured) < 15:
        raise ValueError("Not enough feature-ready observations for a chronological holdout.")

    cut = int(len(featured) * 0.80)
    train, test = featured.iloc[:cut], featured.iloc[cut:]
    if len(test) == 0:
        raise ValueError("The test partition is empty. Add more observations.")

    model = LinearRegression()
    model.fit(train[FEATURES], train[TARGET])
    predicted = model.predict(test[FEATURES])
    actual = test[TARGET].to_numpy(dtype=float)
    naive = test["lag_1"].to_numpy(dtype=float)

    lr_metrics = score(actual, predicted)
    naive_metrics = score(actual, naive)
    result = test[["date", TARGET]].copy()
    result["linear_regression_prediction"] = predicted
    result["naive_last_price_prediction"] = naive
    result["linear_regression_residual"] = actual - predicted
    result.to_csv(output_dir / "test_predictions.csv", index=False)

    # Refit on every feature-ready observation for a next-month demo forecast.
    final_model = LinearRegression().fit(featured[FEATURES], featured[TARGET])
    next_date, forecast_x = next_month_features(df)
    forecast = float(final_model.predict(forecast_x[FEATURES])[0])
    pd.DataFrame([{
        "forecast_month": next_date.strftime("%Y-%m"),
        "predicted_modal_price_rs_per_quintal": forecast,
        "forecast_type": "one-month-ahead demonstration forecast",
        "data_status": "synthetic demonstration data" if False else "depends on input dataset provenance",
    }]).to_csv(output_dir / "next_month_forecast.csv", index=False)
    joblib.dump(final_model, output_dir / "linear_regression_model.joblib")

    metrics = {
        "data_status": data_status,
        "raw_monthly_rows": int(len(df)),
        "feature_ready_rows": int(len(featured)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_period": [train.date.min().strftime("%Y-%m"), train.date.max().strftime("%Y-%m")],
        "test_period": [test.date.min().strftime("%Y-%m"), test.date.max().strftime("%Y-%m")],
        "linear_regression": lr_metrics,
        "naive_last_price_baseline": naive_metrics,
        "next_month_forecast": {
            "month": next_date.strftime("%Y-%m"),
            "predicted_modal_price_rs_per_quintal": forecast,
        },
        "evaluation_protocol": "Chronological 80/20 holdout; one-step-ahead rolling predictions using prior observed prices.",
        "note": "R2 is not a percentage accuracy score. Demo results must not be interpreted as real mandi-market performance.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Coefficients for model interpretation.
    pd.DataFrame({"feature": FEATURES, "coefficient": final_model.coef_}).to_csv(
        output_dir / "model_coefficients.csv", index=False
    )

    # Plots (theme inherited from Matplotlib default, with editorial titles).
    plt.figure(figsize=(10, 4.6))
    plt.plot(df.date, df[TARGET], linewidth=1.8)
    plt.title("Monthly price series | demonstration data")
    plt.xlabel("Month")
    plt.ylabel("₹ per quintal")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "price_trend.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4.6))
    plt.plot(test.date, actual, marker="o", label="Observed")
    plt.plot(test.date, predicted, marker="o", label="Linear Regression")
    plt.plot(test.date, naive, marker=".", linestyle="--", label="Last-price baseline")
    plt.title("Holdout comparison | one-step-ahead predictions")
    plt.xlabel("Month")
    plt.ylabel("₹ per quintal")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.scatter(predicted, actual - predicted, alpha=0.85)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Residual inspection | Linear Regression")
    plt.xlabel("Predicted price (₹/quintal)")
    plt.ylabel("Observed − predicted (₹/quintal)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png", dpi=180)
    plt.close()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a monthly crop-price baseline model.")
    parser.add_argument("--data", default="demo_crop_price_data.csv", help="Input CSV path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--data-status", default=None, help="e.g. SYNTHETIC DEMONSTRATION DATA or VERIFIED AGMARKNET DATA")
    args = parser.parse_args()

    input_path = Path(args.data)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path.resolve()}")
    df = load_monthly_data(input_path)
    status = args.data_status or ("SYNTHETIC DEMONSTRATION DATA - NOT REAL MARKET EVIDENCE" if "demo" in input_path.name.lower() else "PROVENANCE NOT VERIFIED - VERIFY INPUT SOURCE")
    metrics = train_evaluate(df, Path(args.output), data_status=status)
    print("\nCROP PRICE MODEL RUN COMPLETE")
    print(f"Rows: {metrics['raw_monthly_rows']} monthly | {metrics['feature_ready_rows']} feature-ready")
    print(f"Train/test: {metrics['train_rows']}/{metrics['test_rows']}")
    for name in ("linear_regression", "naive_last_price_baseline"):
        m = metrics[name]
        print(f"\n{name.replace('_', ' ').title()}")
        print(f"  MAE : ₹{m['MAE_Rs_per_quintal']:,.2f}/quintal")
        print(f"  RMSE: ₹{m['RMSE_Rs_per_quintal']:,.2f}/quintal")
        print(f"  R²  : {m['R2']:.3f}")
    f = metrics["next_month_forecast"]
    print(f"\nNext-month demo forecast ({f['month']}): ₹{f['predicted_modal_price_rs_per_quintal']:,.2f}/quintal")
    print("IMPORTANT: forecast provenance follows the input file. Synthetic input means this forecast is synthetic too.")
    print(f"Outputs saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
