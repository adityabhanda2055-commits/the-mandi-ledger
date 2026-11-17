"""
THE MANDI LEDGER — MODEL V2
Real-data-ready monthly mandi price forecasting pipeline.

Required historical CSV columns:
    date
    modal_price_rs_per_quintal

Recommended identifier columns:
    commodity, market, variety

Optional numeric explanatory columns:
    arrivals_tonnes, msp_rs_per_quintal, rainfall_mm,
    avg_temperature_c, diesel_price_rs_per_litre,
    fertilizer_price_rs_per_kg, export_restriction_flag,
    import_duty_pct, stock_limit_flag, procurement_flag,
    policy_event_count

Important:
- Historical rows must be genuine recorded observations, not generated/synthetic rows.
- The model uses chronological evaluation; it does not randomly shuffle time-series data.
- External features are lagged by one month in evaluation to reduce leakage.
- A 2027 external-scenario file is optional. Without it, the 2027 forecast uses
  price-history and calendar features only; it does not invent future weather,
  arrivals, MSP or policy decisions.
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =========================
# 1. USER CONFIGURATION
# =========================

# Leave as None when the CSV contains only one commodity / market / variety.
# If multiple series exist, enter the exact value, e.g. "Wheat", "Ludhiana", "Other".
FILTER_COMMODITY = None
FILTER_MARKET = None
FILTER_VARIETY = None

FORECAST_START = pd.Timestamp("2027-01-01")
FORECAST_END = pd.Timestamp("2027-12-01")

PRICE_COLUMN = "modal_price_rs_per_quintal"
DATE_COLUMN = "date"

PRICE_LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]

# These are optional. A field is used only when it exists in the CSV and has
# enough usable historical values. Values must be numeric and source-backed.
OPTIONAL_EXTERNAL_FIELDS = [
    "arrivals_tonnes",
    "msp_rs_per_quintal",
    "rainfall_mm",
    "avg_temperature_c",
    "diesel_price_rs_per_litre",
    "fertilizer_price_rs_per_kg",
    "export_restriction_flag",
    "import_duty_pct",
    "stock_limit_flag",
    "procurement_flag",
    "policy_event_count",
]

# Set to a path such as "/content/future_scenarios_2027.csv" only when you have
# documented monthly scenario inputs for the external fields.
SCENARIO_FILE = None

# In Google Colab, set this True to upload the historical CSV interactively.
COLAB_UPLOAD_HISTORICAL = True


# =========================
# 2. LOAD HISTORICAL DATA
# =========================

def load_historical_csv():
    if COLAB_UPLOAD_HISTORICAL:
        try:
            from google.colab import files
            print("Upload the VERIFIED historical mandi CSV.")
            uploaded = files.upload()
            if not uploaded:
                raise RuntimeError("No file was uploaded.")
            return pd.read_csv(next(iter(uploaded.keys()))), next(iter(uploaded.keys()))
        except ImportError:
            print("Google Colab upload is unavailable. Set COLAB_UPLOAD_HISTORICAL=False and provide a local path.")
    path = input("Enter the path to the historical CSV: ").strip()
    if not path:
        raise ValueError("A historical CSV path is required.")
    return pd.read_csv(path), path


raw, source_file_name = load_historical_csv()
raw.columns = [str(c).strip().lower() for c in raw.columns]

required = {DATE_COLUMN, PRICE_COLUMN}
missing = required - set(raw.columns)
if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}. "
        f"Required columns are '{DATE_COLUMN}' and '{PRICE_COLUMN}'."
    )

raw[DATE_COLUMN] = pd.to_datetime(raw[DATE_COLUMN], errors="coerce")
raw[PRICE_COLUMN] = pd.to_numeric(raw[PRICE_COLUMN], errors="coerce")
raw = raw.dropna(subset=[DATE_COLUMN, PRICE_COLUMN]).copy()
raw = raw[raw[PRICE_COLUMN] > 0].copy()

if raw.empty:
    raise ValueError("No valid positive modal-price observations remain after cleaning.")


# =========================
# 3. SELECT ONE PRICE SERIES
# =========================

def apply_series_filter(frame, column, selected_value):
    if column not in frame.columns:
        return frame

    values = sorted(frame[column].dropna().astype(str).unique().tolist())
    if selected_value is not None:
        return frame[frame[column].astype(str).str.casefold() == str(selected_value).casefold()].copy()

    if len(values) > 1:
        print(f"\nMultiple {column} values found:")
        for value in values:
            print(" -", value)
        choice = input(f"Type the exact {column} to model: ").strip()
        if not choice:
            raise ValueError(f"You must select one {column} to avoid mixing different price series.")
        return frame[frame[column].astype(str).str.casefold() == choice.casefold()].copy()

    return frame


raw = apply_series_filter(raw, "commodity", FILTER_COMMODITY)
raw = apply_series_filter(raw, "market", FILTER_MARKET)
raw = apply_series_filter(raw, "variety", FILTER_VARIETY)

if raw.empty:
    raise ValueError("No rows match the selected commodity / market / variety.")

selected_info = {}
for col in ["commodity", "market", "variety"]:
    selected_info[col] = (
        str(raw[col].dropna().iloc[0]) if col in raw.columns and raw[col].notna().any()
        else "Not specified"
    )

print("\nSelected series:")
for key, value in selected_info.items():
    print(f"{key.title()}: {value}")


# =========================
# 4. AGGREGATE DAILY DATA TO MONTHLY
# =========================

raw["month"] = raw[DATE_COLUMN].dt.to_period("M").dt.to_timestamp()

available_external = [
    c for c in OPTIONAL_EXTERNAL_FIELDS
    if c in raw.columns and pd.to_numeric(raw[c], errors="coerce").notna().sum() >= 18
]

aggregation = {PRICE_COLUMN: "mean"}
for col in available_external:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if col in ["arrivals_tonnes", "rainfall_mm", "policy_event_count"]:
        aggregation[col] = "sum"
    elif col in ["export_restriction_flag", "stock_limit_flag", "procurement_flag"]:
        aggregation[col] = "max"
    else:
        aggregation[col] = "mean"

monthly = raw.groupby("month").agg(aggregation).sort_index()

# Reindex to a continuous monthly calendar. Missing target months remain missing;
# the code does not fabricate or interpolate observed prices.
full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
monthly = monthly.reindex(full_index)
monthly.index.name = "month"

# Ignore any historical rows dated on/after the forecast horizon.
monthly = monthly.loc[monthly.index < FORECAST_START].copy()

if len(monthly) < 30:
    raise ValueError(
        f"Only {len(monthly)} monthly periods are available. "
        "The upgraded model needs at least 30 monthly periods for a meaningful "
        "12-month-lag workflow; more history is strongly preferred."
    )

print(f"\nMonthly periods: {len(monthly)}")
print(f"Historical range: {monthly.index.min().date()} to {monthly.index.max().date()}")
print(f"External numeric fields found: {available_external or 'None'}")


# =========================
# 5. FEATURE ENGINEERING
# =========================

def make_price_features(price_history, target_month, time_index):
    """Build features using only prices observed before target_month."""
    values = np.asarray(price_history, dtype=float)
    row = {}

    for lag in PRICE_LAGS:
        if len(values) < lag or not np.isfinite(values[-lag]):
            row[f"price_lag_{lag}"] = np.nan
        else:
            row[f"price_lag_{lag}"] = float(values[-lag])

    for window in ROLLING_WINDOWS:
        prior = values[-window:] if len(values) >= window else values
        row[f"rolling_mean_{window}"] = (
            float(np.mean(prior)) if len(prior) == window and np.isfinite(prior).all() else np.nan
        )

    month_number = int(target_month.month)
    row["month_sin"] = math.sin(2 * math.pi * month_number / 12)
    row["month_cos"] = math.cos(2 * math.pi * month_number / 12)
    row["time_index"] = int(time_index)
    return row


def build_supervised_frame(data, external_fields=None):
    external_fields = external_fields or []
    prices = data[PRICE_COLUMN].astype(float).tolist()
    rows = []

    for i in range(max(PRICE_LAGS), len(data)):
        target = prices[i]
        if not np.isfinite(target):
            continue

        target_month = data.index[i]
        features = make_price_features(prices[:i], target_month, i)

        # External features are shifted by one monthly observation:
        # the model predicting month t only sees external data from month t-1.
        for col in external_fields:
            previous_value = data[col].iloc[i - 1]
            features[f"{col}_lag1"] = (
                float(previous_value) if pd.notna(previous_value) else np.nan
            )

        rows.append({
            "date": target_month,
            **features,
            "target": float(target),
        })

    result = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return result


price_supervised = build_supervised_frame(monthly)
price_feature_cols = [
    c for c in price_supervised.columns if c != "target"
]
price_supervised = price_supervised.dropna(subset=price_feature_cols + ["target"])

if len(price_supervised) < 24:
    raise ValueError(
        "Not enough complete feature rows remain. Check missing months or provide "
        "a longer historical price series."
    )


# =========================
# 6. CHRONOLOGICAL EVALUATION
# =========================

def regression_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan,
    }


def candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=5.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_candidates(supervised, feature_cols, model_family):
    n = len(supervised)
    test_n = min(12, max(6, int(np.ceil(n * 0.20))))
    split_at = n - test_n

    train = supervised.iloc[:split_at]
    test = supervised.iloc[split_at:]

    if len(train) < 18:
        raise ValueError("Too few training rows after the chronological split.")

    records = []
    fitted = {}

    for name, model in candidate_models().items():
        model.fit(train[feature_cols], train["target"])
        predictions = model.predict(test[feature_cols])
        metrics = regression_metrics(test["target"], predictions)

        records.append({
            "model_family": model_family,
            "model": name,
            "test_rows": len(test),
            **metrics,
        })
        fitted[name] = model

    scores = pd.DataFrame(records).sort_values("MAE").reset_index(drop=True)
    best_name = scores.loc[0, "model"]
    return scores, fitted[best_name], train, test, best_name


price_scores, best_price_model, price_train, price_test, best_price_name = (
    evaluate_candidates(price_supervised, price_feature_cols, "Price + Calendar")
)

# Seasonal-naive benchmark: predict this month using the observed price 12 months earlier.
seasonal_actual = []
seasonal_predicted = []
for date in price_test.index:
    position = monthly.index.get_loc(date)
    if position >= 12:
        actual_value = monthly.loc[date, PRICE_COLUMN]
        prior_value = monthly.iloc[position - 12][PRICE_COLUMN]
        if pd.notna(actual_value) and pd.notna(prior_value):
            seasonal_actual.append(float(actual_value))
            seasonal_predicted.append(float(prior_value))

if len(seasonal_actual) >= 2:
    seasonal_metrics = regression_metrics(seasonal_actual, seasonal_predicted)
    print("\nSeasonal-naive baseline (same month, previous year):")
    print(seasonal_metrics)

print("\nChronological test-set model comparison:")
print(price_scores.to_string(index=False))
print(f"\nSelected price-only model: {best_price_name}")


# =========================
# 7. OPTIONAL EXTERNAL-FACTOR MODEL
# =========================

# This model is evaluated only when the historical data contains numeric external
# variables and a future-scenario CSV is supplied. The scenario file must contain
# date plus the same numeric external columns for the forecast period.
best_model = best_price_model
best_model_name = best_price_name
best_feature_cols = price_feature_cols
best_family = "Price + Calendar"
best_scores = price_scores.copy()
external_model_used = False
scenario = None

if SCENARIO_FILE:
    scenario = pd.read_csv(SCENARIO_FILE)
    scenario.columns = [str(c).strip().lower() for c in scenario.columns]
    if DATE_COLUMN not in scenario.columns:
        raise ValueError("Scenario CSV must contain a 'date' column.")

    scenario[DATE_COLUMN] = pd.to_datetime(scenario[DATE_COLUMN], errors="coerce")
    scenario = scenario.dropna(subset=[DATE_COLUMN]).copy()
    scenario["month"] = scenario[DATE_COLUMN].dt.to_period("M").dt.to_timestamp()
    scenario = scenario.set_index("month").sort_index()

    usable_external = [
        c for c in available_external
        if c in scenario.columns
        and pd.to_numeric(scenario[c], errors="coerce").notna().sum() >= 6
    ]

    if usable_external:
        for col in usable_external:
            scenario[col] = pd.to_numeric(scenario[col], errors="coerce")

        external_supervised = build_supervised_frame(monthly, usable_external)
        external_feature_cols = [
            c for c in external_supervised.columns if c != "target"
        ]
        external_supervised = external_supervised.dropna(
            subset=external_feature_cols + ["target"]
        )

        if len(external_supervised) >= 24:
            ext_scores, ext_model, ext_train, ext_test, ext_name = evaluate_candidates(
                external_supervised, external_feature_cols, "Price + Lagged External Factors"
            )
            print("\nExternal-factor model comparison:")
            print(ext_scores.to_string(index=False))

            # Only choose the external model if its chronological test MAE is lower.
            price_best_mae = float(price_scores.iloc[0]["MAE"])
            ext_best_mae = float(ext_scores.iloc[0]["MAE"])

            if ext_best_mae < price_best_mae:
                required_future_months = pd.date_range(
                    monthly.index.max() + pd.offsets.MonthBegin(1),
                    FORECAST_END,
                    freq="MS",
                )
                missing_scenario_months = [
                    d for d in required_future_months
                    if d > monthly.index.max() and d not in scenario.index
                ]
                missing_scenario_values = [
                    c for c in usable_external
                    if c not in scenario.columns
                ]

                if not missing_scenario_months and not missing_scenario_values:
                    best_model = ext_model
                    best_model_name = ext_name
                    best_feature_cols = external_feature_cols
                    best_family = "Price + Lagged External Factors"
                    best_scores = ext_scores
                    external_model_used = True
                else:
                    print(
                        "\nExternal model had a lower test MAE, but its scenario file "
                        "does not cover every required future month. Using price-only "
                        "model for the 2027 forecast."
                    )
            else:
                print("\nExternal factors did not improve holdout MAE; retaining price-only model.")
        else:
            print("\nNot enough complete rows for a reliable external-factor model.")
    else:
        print("\nNo usable external factors found in both historical and scenario files.")
else:
    print(
        "\nNo future-scenario CSV supplied. The forecast will use price history and "
        "calendar features only; future weather, arrivals and policy decisions will "
        "not be invented."
    )


# =========================
# 8. REFIT SELECTED MODEL ON ALL AVAILABLE HISTORY
# =========================

if external_model_used:
    final_supervised = build_supervised_frame(monthly, usable_external)
    final_feature_cols = [c for c in final_supervised.columns if c != "target"]
    final_supervised = final_supervised.dropna(subset=final_feature_cols + ["target"])
else:
    final_supervised = price_supervised.copy()
    final_feature_cols = price_feature_cols

final_model = candidate_models()[best_model_name] if best_family == "Price + Calendar" else candidate_models()[best_model_name]
final_model.fit(final_supervised[final_feature_cols], final_supervised["target"])


# =========================
# 9. RECURSIVE FORECAST THROUGH DECEMBER 2027
# =========================

def latest_external_values(data, fields):
    latest = {}
    for col in fields:
        valid = data[col].dropna()
        latest[col] = float(valid.iloc[-1]) if len(valid) else np.nan
    return latest


history_prices = monthly[PRICE_COLUMN].astype(float).tolist()
history_dates = list(monthly.index)
external_history = latest_external_values(monthly, usable_external) if external_model_used else {}

# Scenario values are interpreted as dated assumptions/forecasts, not observations.
scenario_by_month = scenario.to_dict(orient="index") if external_model_used else {}

next_month = monthly.index.max() + pd.offsets.MonthBegin(1)
forecast_months = pd.date_range(next_month, FORECAST_END, freq="MS")

forecast_records = []
for step, forecast_month in enumerate(forecast_months):
    features = make_price_features(
        history_prices,
        forecast_month,
        len(history_prices),
    )

    if external_model_used:
        # For the prediction at month t, use the latest known external values
        # from the prior month. After forecasting t, advance to scenario values
        # for t so they can be used for t+1.
        for col in usable_external:
            value = external_history.get(col, np.nan)
            features[f"{col}_lag1"] = value

    x_future = pd.DataFrame([features])[final_feature_cols]

    if x_future.isna().any().any():
        raise ValueError(
            f"Missing feature values while forecasting {forecast_month:%Y-%m}. "
            "Check the 2027 scenario CSV or provide more continuous historical data."
        )

    prediction = float(final_model.predict(x_future)[0])
    forecast_records.append({
        "date": forecast_month.strftime("%Y-%m-%d"),
        "predicted_modal_price_rs_per_quintal": prediction,
        "forecast_type": "MODEL_FORECAST",
        "model": best_model_name,
        "model_family": best_family,
    })

    history_prices.append(prediction)
    history_dates.append(forecast_month)

    if external_model_used and forecast_month in scenario_by_month:
        for col in usable_external:
            next_value = scenario_by_month[forecast_month].get(col, np.nan)
            if pd.notna(next_value):
                external_history[col] = float(next_value)

forecast_all = pd.DataFrame(forecast_records)
forecast_2027 = forecast_all[
    (pd.to_datetime(forecast_all["date"]) >= FORECAST_START)
    & (pd.to_datetime(forecast_all["date"]) <= FORECAST_END)
].copy()

# Empirical error band from the selected model's chronological test residuals.
# This is a historical-error range, not a formal probabilistic confidence interval.
if external_model_used:
    eval_supervised = build_supervised_frame(monthly, usable_external)
    eval_features = [c for c in eval_supervised.columns if c != "target"]
    eval_supervised = eval_supervised.dropna(subset=eval_features + ["target"])
    _, eval_model, _, eval_test, _ = evaluate_candidates(
        eval_supervised, eval_features, "Price + Lagged External Factors"
    )
    eval_split = len(eval_supervised) - len(eval_test)
    eval_train = eval_supervised.iloc[:eval_split]
    eval_model.fit(eval_train[eval_features], eval_train["target"])
    eval_pred = eval_model.predict(eval_test[eval_features])
    residuals = eval_test["target"].to_numpy() - eval_pred
else:
    eval_model = candidate_models()[best_price_name]
    n = len(price_supervised)
    test_n = min(12, max(6, int(np.ceil(n * 0.20))))
    eval_train = price_supervised.iloc[:-test_n]
    eval_test = price_supervised.iloc[-test_n:]
    eval_model.fit(eval_train[price_feature_cols], eval_train["target"])
    eval_pred = eval_model.predict(eval_test[price_feature_cols])
    residuals = eval_test["target"].to_numpy() - eval_pred

lower_error = float(np.quantile(residuals, 0.10))
upper_error = float(np.quantile(residuals, 0.90))
forecast_2027["historical_error_band_low"] = (
    forecast_2027["predicted_modal_price_rs_per_quintal"] + lower_error
).clip(lower=0)
forecast_2027["historical_error_band_high"] = (
    forecast_2027["predicted_modal_price_rs_per_quintal"] + upper_error
).clip(lower=0)

# Keep only the forecast output rows for calendar year 2027.
forecast_2027.to_csv("mandi_ledger_forecast_2027.csv", index=False)

# Save scores and trained model package.
best_scores.to_csv("mandi_ledger_model_scores.csv", index=False)
model_bundle = {
    "model": final_model,
    "model_name": best_model_name,
    "model_family": best_family,
    "feature_columns": final_feature_cols,
    "price_column": PRICE_COLUMN,
    "history_start": str(monthly.index.min().date()),
    "history_end": str(monthly.index.max().date()),
    "commodity": selected_info["commodity"],
    "market": selected_info["market"],
    "variety": selected_info["variety"],
    "external_model_used": external_model_used,
    "external_fields": usable_external if external_model_used else [],
    "forecast_year": 2027,
    "note": "Forecast values are estimates, not observed mandi prices.",
}
joblib.dump(model_bundle, "mandi_ledger_model_v2.joblib")


# =========================
# 10. RESULTS AND VISUALIZATION
# =========================

print("\n================ MODEL SUMMARY ================")
print("Selected model:", best_model_name)
print("Model family:", best_family)
print("Historical data range:", monthly.index.min().date(), "to", monthly.index.max().date())
print("Historical monthly rows:", len(monthly))
print("External factors used for 2027 forecast:", external_model_used)
print("\n2027 monthly forecast (₹/quintal):")
print(forecast_2027.to_string(index=False))

plt.figure(figsize=(12, 5))
plt.plot(monthly.index, monthly[PRICE_COLUMN], label="Observed monthly modal price")
plt.plot(
    pd.to_datetime(forecast_2027["date"]),
    forecast_2027["predicted_modal_price_rs_per_quintal"],
    marker="o",
    label="2027 model forecast",
)
plt.fill_between(
    pd.to_datetime(forecast_2027["date"]),
    forecast_2027["historical_error_band_low"],
    forecast_2027["historical_error_band_high"],
    alpha=0.18,
    label="Historical-error range (not a formal confidence interval)",
)
plt.title(f"Crop Price Forecast — {selected_info['commodity']} / {selected_info['market']}")
plt.xlabel("Month")
plt.ylabel("Modal price (₹ per quintal)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("mandi_ledger_forecast_2027.png", dpi=180)
plt.show()

print("\nCreated files:")
print("1. mandi_ledger_forecast_2027.csv")
print("2. mandi_ledger_model_scores.csv")
print("3. mandi_ledger_model_v2.joblib")
print("4. mandi_ledger_forecast_2027.png")

# Download outputs automatically in Google Colab.
try:
    from google.colab import files
    for output_file in [
        "mandi_ledger_forecast_2027.csv",
        "mandi_ledger_model_scores.csv",
        "mandi_ledger_model_v2.joblib",
        "mandi_ledger_forecast_2027.png",
    ]:
        files.download(output_file)
except ImportError:
    print("Files are saved in the current working directory.")
