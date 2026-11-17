from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "modal_price_rs_per_quintal"

FEATURES = [
    "month",
    "year",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6"
]

FORECAST_END = pd.Timestamp("2027-12-01")


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

MODEL_FACTORIES = {
    "Linear Regression": lambda: LinearRegression(),

    "Ridge Regression": lambda: make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=7,
        l2_regularization=1.0,
        random_state=42
    )
}


# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------

def normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns, candidates):

    lookup = {
        normalize_name(column): column
        for column in columns
    }

    for candidate in candidates:
        found = lookup.get(normalize_name(candidate))

        if found is not None:
            return found

    return None


def load_monthly_data(path_or_buffer):

    raw = pd.read_csv(path_or_buffer)

    if raw.empty:
        raise ValueError("The CSV contains no records.")

    date_column = find_column(
        raw.columns,
        ["date", "arrival_date", "arrival date"]
    )

    price_column = find_column(
        raw.columns,
        [
            TARGET,
            "modal_price",
            "modal price",
            "modal price (rs/quintal)",
            "modal price (₹/quintal)"
        ]
    )

    if date_column is None or price_column is None:

        raise ValueError(
            "Could not identify the date and modal-price columns. "
            "Expected date + modal_price_rs_per_quintal "
            "or arrival_date + modal_price."
        )

    # Prevent mixing multiple crops or markets.

    identifier_aliases = {
        "state",
        "statename",
        "district",
        "districtname",
        "market",
        "marketname",
        "commodity",
        "commodityname",
        "variety",
        "crop",
        "cropname"
    }

    for column in raw.columns:

        if normalize_name(column) in identifier_aliases:

            values = (
                raw[column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if values.nunique() > 1:

                examples = ", ".join(
                    values.drop_duplicates().head(3).tolist()
                )

                raise ValueError(
                    f"Column '{column}' contains multiple values "
                    f"({examples}). Filter the CSV to one crop, "
                    "one market, one location and one variety."
                )

    date_text = raw[date_column].astype(str).str.strip()

    iso_mask = date_text.str.match(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    )

    parsed_dates = pd.Series(
        pd.NaT,
        index=raw.index,
        dtype="datetime64[ns]"
    )

    parsed_dates.loc[iso_mask] = pd.to_datetime(
        date_text.loc[iso_mask],
        errors="coerce",
        dayfirst=False
    )

    parsed_dates.loc[~iso_mask] = pd.to_datetime(
        date_text.loc[~iso_mask],
        errors="coerce",
        dayfirst=True
    )

    df = pd.DataFrame({
        "date": parsed_dates,

        TARGET: pd.to_numeric(
            raw[price_column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce"
        )
    })
    df = df.dropna(subset=["date", TARGET]).copy()

    if df.empty:
        raise ValueError("No valid date and modal-price rows found.")

    if (df[TARGET] <= 0).any():
        raise ValueError("Modal prices must be positive.")

    # Aggregate daily prices into monthly observations.

    df["date"] = (
        df["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    df = (
        df.groupby("date", as_index=False)[TARGET]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Check missing months.

    expected = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="MS"
    )

    missing = expected.difference(
        pd.DatetimeIndex(df["date"])
    )

    if len(missing):

        sample = ", ".join(
            d.strftime("%Y-%m")
            for d in missing[:8]
        )

        raise ValueError(
            f"Missing months detected: {sample}. "
            "Choose a continuous date range. "
            "The program will not invent missing prices."
        )

    if len(df) < 36:

        raise ValueError(
            "At least 36 consecutive monthly observations "
            "are recommended."
        )

    return df


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def make_features(df):

    out = (
        df.copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year

    for lag in (1, 2, 3, 12):

        out[f"lag_{lag}"] = (
            out[TARGET].shift(lag)
        )

    out["rolling_mean_3"] = (
        out[TARGET]
        .shift(1)
        .rolling(3)
        .mean()
    )

    out["rolling_mean_6"] = (
        out[TARGET]
        .shift(1)
        .rolling(6)
        .mean()
    )

    return (
        out.dropna()
        .reset_index(drop=True)
    )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def score(actual, predicted):

    return {

        "MAE_Rs_per_quintal": float(
            mean_absolute_error(actual, predicted)
        ),

        "RMSE_Rs_per_quintal": float(
            math.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),

        "R2": float(
            r2_score(actual, predicted)
        )

    }


# --------------------------------------------------
# FUTURE FEATURE GENERATION
# --------------------------------------------------

def make_future_feature_row(date, history):

    def value_at(months_ago):

        key = date - pd.DateOffset(
            months=months_ago
        )

        if key not in history:

            raise ValueError(
                f"Missing historical value for {date:%Y-%m}."
            )

        return float(history[key])

    row = {

        "month": date.month,
        "year": date.year,

        "lag_1": value_at(1),
        "lag_2": value_at(2),
        "lag_3": value_at(3),
        "lag_12": value_at(12),

        "rolling_mean_3": np.mean([
            value_at(1),
            value_at(2),
            value_at(3)
        ]),

        "rolling_mean_6": np.mean([
            value_at(i)
            for i in range(1, 7)
        ])

    }

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


# --------------------------------------------------
# FORECASTING THROUGH 2027
# --------------------------------------------------

def forecast_through_2027(
    model,
    df,
    forecast_end=FORECAST_END
):

    last_date = pd.Timestamp(
        df["date"].max()
    )

    if last_date >= forecast_end:

        return pd.DataFrame(columns=[
            "forecast_month",
            "predicted_modal_price_rs_per_quintal",
            "forecast_type"
        ])

    history = {

        pd.Timestamp(row.date): float(
            getattr(row, TARGET)
        )

        for row in df[
            ["date", TARGET]
        ].itertuples(index=False)

    }

    future_dates = pd.date_range(

        last_date + pd.offsets.MonthBegin(1),

        forecast_end,

        freq="MS"

    )

    forecast_rows = []

    for date in future_dates:

        features = make_future_feature_row(
            date,
            history
        )

        prediction = float(
            model.predict(
                features[FEATURES]
            )[0]
        )

        # Prevent negative displayed prices.

        prediction = max(0.0, prediction)

        # Use each predicted value for the next month.

        history[pd.Timestamp(date)] = prediction

        forecast_rows.append({

            "forecast_month": date.strftime("%Y-%m"),

            "predicted_modal_price_rs_per_quintal": prediction,

            "forecast_type": "recursive model forecast"

        })

    return pd.DataFrame(forecast_rows)


# --------------------------------------------------
# TRAINING AND COMPLETE EVALUATION
# --------------------------------------------------

def train_evaluate(
    df,
    output_dir,
    data_status="PROVENANCE NOT VERIFIED"
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    featured = make_features(df)

    if len(featured) < 18:

        raise ValueError(
            "Not enough feature-ready observations."
        )

    # Chronological 80/20 split.

    split = int(len(featured) * 0.80)

    train = featured.iloc[:split]
    test = featured.iloc[split:]

    if len(test) < 2:

        raise ValueError(
            "The chronological test partition is too small."
        )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET].to_numpy(dtype=float)

    comparison = []
    fitted_holdout = {}

    # Train and compare four ML models.

    for name, factory in MODEL_FACTORIES.items():

        estimator = factory()

        estimator.fit(
            X_train,
            y_train
        )

        prediction = estimator.predict(X_test)

        fitted_holdout[name] = (
            estimator,
            prediction
        )

        comparison.append({

            "model": name,

            **score(
                y_test,
                prediction
            )

        })

    # Baseline models.

    last_price = (
        test["lag_1"]
        .to_numpy(dtype=float)
    )

    seasonal_naive = (
        test["lag_12"]
        .to_numpy(dtype=float)
    )

    comparison.extend([

        {
            "model": "Last-price baseline",
            **score(y_test, last_price)
        },

        {
            "model": "Seasonal-naive (12-month)",
            **score(y_test, seasonal_naive)
        }

    ])

    comparison_df = pd.DataFrame(comparison)

    comparison_df = comparison_df.sort_values(
        "MAE_Rs_per_quintal",
        ascending=True
    )

    comparison_df.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    # Select best ML model using holdout MAE.

    ml_results = comparison_df[
        comparison_df["model"].isin(
            MODEL_FACTORIES.keys()
        )
    ]

    best_row = ml_results.sort_values(
        "MAE_Rs_per_quintal"
    ).iloc[0]

    best_name = str(best_row["model"])

    best_prediction = (
        fitted_holdout[best_name][1]
    )

    # Retain Linear Regression metrics for old app.py compatibility.

    linear_prediction = (
        fitted_holdout["Linear Regression"][1]
    )

    linear_metrics = score(
        y_test,
        linear_prediction
    )

    last_metrics = score(
        y_test,
        last_price
    )

    seasonal_metrics = score(
        y_test,
        seasonal_naive
    )

    best_metrics = score(
        y_test,
        best_prediction
    )

    # Save holdout predictions.

    prediction_frame = test[
        ["date", TARGET]
    ].copy()

    prediction_frame[
        "linear_regression_prediction"
    ] = linear_prediction

    prediction_frame[
        "best_model_prediction"
    ] = best_prediction

    prediction_frame[
        "best_model"
    ] = best_name

    prediction_frame[
        "naive_last_price_prediction"
    ] = last_price

    prediction_frame[
        "seasonal_naive_prediction"
    ] = seasonal_naive

    prediction_frame[
        "best_model_residual"
    ] = y_test - best_prediction

    prediction_frame.to_csv(
        output_dir / "test_predictions.csv",
        index=False
    )

    # Refit the selected model using all available data.

    final_model = MODEL_FACTORIES[
        best_name
    ]()

    final_model.fit(
        featured[FEATURES],
        featured[TARGET]
    )

    joblib.dump(
        final_model,
        output_dir / "best_price_model.joblib"
    )

    # Save a separate Linear Regression model.

    linear_final = LinearRegression()

    linear_final.fit(
        featured[FEATURES],
        featured[TARGET]
    )

    joblib.dump(
        linear_final,
        output_dir / "linear_regression_model.joblib"
    )

    # Generate forecast through December 2027.

    forecast_df = forecast_through_2027(
        final_model,
        df
    )

    forecast_df.insert(
        0,
        "model",
        best_name
    )

    forecast_df.insert(
        1,
        "data_status",
        data_status
    )

    forecast_df.to_csv(
        output_dir / "forecast_to_2027.csv",
        index=False
    )

    if len(forecast_df):

        first = forecast_df.iloc[0]

        next_forecast = {

            "month": str(
                first["forecast_month"]
            ),

            "predicted_modal_price_rs_per_quintal": float(
                first[
                    "predicted_modal_price_rs_per_quintal"
                ]
            )

        }

    else:

        next_forecast = {

            "month": None,

            "predicted_modal_price_rs_per_quintal": None

        }

    # Metrics report.

    metrics = {

        "data_status": data_status,

        "raw_monthly_rows": int(len(df)),

        "feature_ready_rows": int(len(featured)),

        "train_rows": int(len(train)),

        "test_rows": int(len(test)),

        "train_period": [

            train.date.min().strftime("%Y-%m"),

            train.date.max().strftime("%Y-%m")

        ],

        "test_period": [

            test.date.min().strftime("%Y-%m"),

            test.date.max().strftime("%Y-%m")

        ],

        "linear_regression": linear_metrics,

        "naive_last_price_baseline": last_metrics,

        "seasonal_naive_baseline": seasonal_metrics,

        "best_model": best_name,

        "best_model_metrics": best_metrics,

        "model_comparison": comparison,

        "next_month_forecast": next_forecast,

        "forecast_end": FORECAST_END.strftime("%Y-%m"),

        "forecast_months": int(len(forecast_df)),

        "evaluation_protocol": (
            "Chronological 80/20 holdout; "
            "one-step-ahead rolling predictions."
        ),

        "forecast_protocol": (
            "Recursive monthly forecasting. "
            "Each predicted month becomes an input "
            "for subsequent months."
        ),

        "note": (
            "R2 is not percentage accuracy. "
            "Forecasts are estimates, not actual prices. "
            "Synthetic input produces synthetic results."
        )

    }

    (output_dir / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------

    # 1. Historical price trend.

    plt.figure(figsize=(11, 5))

    plt.plot(
        df["date"],
        df[TARGET],
        linewidth=1.8,
        label="Observed monthly price"
    )

    plt.title("Monthly Modal Price History")
    plt.xlabel("Month")
    plt.ylabel("Price (₹ per quintal)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(
        output_dir / "price_trend.png",
        dpi=180
    )

    plt.close()

    # 2. Actual versus predicted.

    plt.figure(figsize=(11, 5))

    plt.plot(
        test["date"],
        y_test,
        marker="o",
        label="Observed"
    )

    plt.plot(
        test["date"],
        best_prediction,
        marker="o",
        label=f"Best model: {best_name}"
    )

    plt.plot(
        test["date"],
        last_price,
        linestyle="--",
        label="Last-price baseline"
    )

    plt.title(
        "Chronological Holdout: Observed vs Predicted"
    )

    plt.xlabel("Month")
    plt.ylabel("Price (₹ per quintal)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(
        output_dir / "actual_vs_predicted.png",
        dpi=180
    )

    plt.close()

    # 3. Residual analysis.

    plt.figure(figsize=(8, 5))

    plt.scatter(
        best_prediction,
        y_test - best_prediction,
        alpha=0.8
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        f"Residual Inspection: {best_name}"
    )

    plt.xlabel("Predicted price (₹ per quintal)")
    plt.ylabel("Observed - Predicted")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "residuals.png",
        dpi=180
    )

    plt.close()

    # 4. Forecast visualization.

    plt.figure(figsize=(11, 5))

    recent = df.tail(
        min(36, len(df))
    )

    plt.plot(
        recent["date"],
        recent[TARGET],
        marker=".",
        label="Observed"
    )

    if len(forecast_df):

        future_dates = pd.to_datetime(
            forecast_df["forecast_month"]
        )

        future_values = forecast_df[
            "predicted_modal_price_rs_per_quintal"
        ]

        plt.plot(
            future_dates,
            future_values,
            marker="o",
            linestyle="--",
            label="Forecast"
        )

        plt.axvline(
            df["date"].max(),
            linestyle=":",
            linewidth=1.5,
            label="Forecast starts"
        )

    plt.title(
        f"Price Forecast Through {FORECAST_END:%B %Y}"
    )

    plt.xlabel("Month")
    plt.ylabel("Price (₹ per quintal)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(
        output_dir / "forecast_to_2027.png",
        dpi=180
    )

    plt.close()

    return metrics


# --------------------------------------------------
# COMMAND LINE EXECUTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Crop price model comparison and "
            "forecasting through December 2027."
        )
    )

    parser.add_argument(
        "--data",
        default="demo_crop_price_data.csv",
        help="Input CSV file"
    )

    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory"
    )

    parser.add_argument(
        "--data-status",
        default=None,
        help="Description of input data provenance"
    )

    args = parser.parse_args()

    input_path = Path(args.data)

    if not input_path.exists():

        raise SystemExit(
            f"Input file not found: {input_path.resolve()}"
        )

    df = load_monthly_data(
        input_path
    )

    if args.data_status:

        status = args.data_status

    elif "demo" in input_path.name.lower():

        status = (
            "SYNTHETIC DEMONSTRATION DATA "
            "- NOT REAL MARKET EVIDENCE"
        )

    else:

        status = (
            "UPLOADED CSV "
            "- SOURCE MUST BE VERIFIED BY USER"
        )

    metrics = train_evaluate(
        df,
        Path(args.output),
        data_status=status
    )

    print("\nTHE MANDI LEDGER - MODEL RUN COMPLETE")

    print(
        f"Monthly observations: "
        f"{metrics['raw_monthly_rows']}"
    )

    print(
        f"Training/testing rows: "
        f"{metrics['train_rows']}/"
        f"{metrics['test_rows']}"
    )

    print(
        f"Best ML model: "
        f"{metrics['best_model']}"
    )

    print("\nMODEL COMPARISON")

    comparison = pd.DataFrame(
        metrics["model_comparison"]
    )

    print(
        comparison.sort_values(
            "MAE_Rs_per_quintal"
        ).to_string(index=False)
    )

    print(
        f"\nForecast months generated: "
        f"{metrics['forecast_months']}"
    )

    print(
        f"Forecast ends: "
        f"{metrics['forecast_end']}"
    )

    print(
        "\nIMPORTANT: Forecast values are estimates, "
        "not actual mandi prices."
    )

    print(
        f"Outputs saved to: "
        f"{Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
