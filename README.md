# The Mandi Ledger — Crop Price Prediction Model

A reproducible educational machine-learning project using a Linear Regression baseline.

## Important data note

`demo_crop_price_data.csv` is **synthetic demonstration data**. It is not downloaded from AGMARKNET and it is not evidence of actual mandi prices. Metrics and the sample next-month estimate generated from it are demonstrations only.

For a real empirical run, obtain data from the Government of India's Open Government Data Platform / AGMARKNET catalogue: https://data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi. The catalogue describes daily wholesale minimum, maximum and modal prices generated through AGMARKNET. Filter to **one crop, one market, one variety (where relevant), and one unit**, then prepare a continuous monthly series. The loader aggregates multiple daily records within a month by taking the mean of their daily modal prices. If source identifier columns are retained, the code rejects columns containing multiple values rather than silently combining different markets or crops. It does not fill missing months or invent prices.

## Files

- `crop_price_model.py` — data validation, feature engineering, chronological holdout, Linear Regression, naïve baseline, metrics, plots, next-month forecast, model artifact.
- `app.py` — optional Streamlit front end.
- `demo_crop_price_data.csv` — synthetic 96-month demo series.
- `requirements.txt` — Python dependencies.
- `outputs/` — generated after running the model.

## Run the model (Windows)

1. Install Python 3.10 or later.
2. Open Command Prompt in this project folder.
3. Create and activate an environment:

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bat
   python -m pip install -r requirements.txt
   ```

5. Train and evaluate:

   ```bat
   python crop_price_model.py --data demo_crop_price_data.csv --output outputs
   ```

Generated files include metrics, holdout predictions, coefficient values, three charts, a saved model, and the next-month demo forecast.

## Run the interactive app

```bat
streamlit run app.py
```

The app opens in a browser. Upload your cleaned monthly CSV to run the same workflow on your selected series. The included CSV is selected by default when no file is uploaded.

## CSV format

Required columns:

```csv
date,modal_price_rs_per_quintal
2018-01-01,1842.29
2018-02-01,1865.49
```

- `date`: observation date; daily records can be supplied, then they are aggregated to month.
- `modal_price_rs_per_quintal`: modal wholesale price in ₹ per quintal.
- Use one crop, one market, one variety, and one consistent unit per model run. If identifier columns such as `Market` or `Commodity` are present, the program checks that each contains only one non-empty value.
- The resulting monthly series must have no missing months. The program raises an error rather than silently filling gaps.

## Method

Features: month, year, previous month price (`lag_1`), two-month lag (`lag_2`), twelve-month lag (`lag_12`), and the mean of the previous three monthly observations (`rolling_mean_3`). All price-derived features are shifted so the current target is not used to predict itself.

The data is sorted by time and split chronologically: earliest 80% for training and latest 20% for testing. Holdout predictions are **one-step-ahead rolling predictions**: when predicting a test month, earlier observed monthly prices are assumed to be available. This is not a multi-month recursive forecast.

A naïve baseline predicts the next month's price as the last observed price. Compare Linear Regression with this baseline before claiming the model adds value.

## Limitations

This baseline uses only historical price and calendar features. It does not include weather, arrivals, production, transport, storage, policy, imports/exports, or market-specific shocks. A one-split demonstration is not a production-grade validation. Use rolling-origin backtesting and verified source data before making real-world claims.

## Suggested next improvement

After validating the pipeline on official data, compare Linear Regression with seasonal-naïve, Random Forest, and a time-series model using the same date-aware evaluation protocol. Keep a separate test period untouched until final model selection.
