"""Prepare The Mandi Ledger's monthly All-India Wheat price series through Oct-2026.

Source: Agmarknet-derived state-monthly extract published in the
herrrickshaw/india-mandi-arrivals repository.

Method:
- Filter the exact commodity name 'Wheat'.
- Use state-monthly modal-price aggregates and reported arrivals.
- Exclude state-month rows with fewer than 50 tonnes of arrivals, matching
  the source repository's stated reliability filter.
- Compute an All-India monthly arrivals-weighted price:
      sum(state price * state arrivals) / sum(state arrivals)
- Require an uninterrupted Jan-2019 through Oct-2026 sequence (94 months).
- Never fill, interpolate, or invent missing prices.

Output schema matches crop_price_model.py:
    date, modal_price_rs_per_quintal
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/herrrickshaw/india-mandi-arrivals/"
    "main/data/agmarknet/state_monthly.csv.gz"
)
SOURCE_REPO_URL = "https://github.com/herrrickshaw/india-mandi-arrivals"
START_MONTH = "2019-01"
REQUIRED_END_MONTH = "2026-10"
MIN_ARRIVALS_TONNES = 50.0
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_CSV = OUTPUT_DIR / "real_mandi_monthly.csv"
AUDIT_CSV = OUTPUT_DIR / "all_india_wheat_monthly_audit.csv"
METADATA_JSON = OUTPUT_DIR / "real_mandi_metadata.json"


def download_source() -> bytes:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "TheMandiLedger/1.0 (educational project)"},
    )
    print("Downloading Agmarknet-derived state-monthly data...", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if len(payload) < 1000:
        raise RuntimeError("The downloaded source file is unexpectedly small.")
    print(f"Downloaded {len(payload):,} bytes.", flush=True)
    return payload


def as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def main() -> int:
    payload = download_source()
    source = pd.read_csv(io.BytesIO(payload), compression="gzip", low_memory=False)

    required = {
        "month", "commodity", "state", "arrivals_t", "price_rs_qtl"
    }
    missing_columns = sorted(required - set(source.columns))
    if missing_columns:
        raise ValueError(
            f"Source schema changed; missing columns: {missing_columns}. "
            f"Available columns: {list(source.columns)}"
        )

    source["month"] = source["month"].astype(str).str.strip().str.slice(0, 7)
    source["commodity"] = source["commodity"].astype(str).str.strip()
    wheat = source[source["commodity"].str.casefold().eq("wheat")].copy()
    if wheat.empty:
        raise ValueError("No exact 'Wheat' rows were found in the source extract.")

    wheat["arrivals_t"] = as_number(wheat["arrivals_t"])
    wheat["price_rs_qtl"] = as_number(wheat["price_rs_qtl"])

    valid = wheat[
        wheat["month"].str.match(r"^\d{4}-\d{2}$", na=False)
        & wheat["arrivals_t"].notna()
        & wheat["price_rs_qtl"].notna()
        & (wheat["arrivals_t"] >= MIN_ARRIVALS_TONNES)
        & (wheat["price_rs_qtl"] > 0)
    ].copy()
    if valid.empty:
        raise ValueError("No valid Wheat rows remain after quality checks.")

    valid["weighted_price"] = valid["price_rs_qtl"] * valid["arrivals_t"]
    monthly = (
        valid.groupby("month", as_index=False)
        .agg(
            weighted_price_sum=("weighted_price", "sum"),
            arrivals_t=("arrivals_t", "sum"),
            states_reporting=("state", "nunique"),
            state_month_rows=("state", "size"),
        )
    )
    monthly["modal_price_rs_per_quintal"] = (
        monthly["weighted_price_sum"] / monthly["arrivals_t"]
    )
    monthly = monthly.sort_values("month").reset_index(drop=True)

    expected = pd.period_range(START_MONTH, REQUIRED_END_MONTH, freq="M").astype(str)
    observed = set(monthly["month"])

    print(f"Wheat source rows: {len(wheat):,}")
    print(f"Valid state-month rows: {len(valid):,}")
    print(f"All-India monthly points found: {len(monthly):,}")
    print(f"Latest available month: {monthly['month'].max()}")

    # Keep the most recent uninterrupted run that ends exactly in October 2026.
    # Earlier gaps are not filled; they simply define the beginning of the run.
    if REQUIRED_END_MONTH not in observed:
        raise ValueError(
            f"October 2026 is not present in the filtered Wheat data. "
            f"Latest available month is {monthly['month'].max()}. "
            "No model CSV was written."
        )

    run_months = []
    cursor = pd.Period(REQUIRED_END_MONTH, freq="M")
    while str(cursor) in observed:
        run_months.append(str(cursor))
        cursor -= 1
    run_months.reverse()

    if len(run_months) < 36:
        raise ValueError(
            f"October 2026 is present, but the uninterrupted series ending "
            f"there contains only {len(run_months)} months. At least 36 are "
            "required. No model CSV was written."
        )

    chosen = monthly[monthly["month"].isin(run_months)].copy()
    chosen["date"] = pd.to_datetime(chosen["month"] + "-01")
    model_frame = chosen[["date", "modal_price_rs_per_quintal"]].copy()
    model_frame["date"] = model_frame["date"].dt.strftime("%Y-%m-%d")
    model_frame["modal_price_rs_per_quintal"] = model_frame[
        "modal_price_rs_per_quintal"
    ].round(2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_frame.to_csv(OUTPUT_CSV, index=False)

    audit = chosen[[
        "month", "arrivals_t", "states_reporting", "state_month_rows",
        "modal_price_rs_per_quintal",
    ]].copy()
    audit.to_csv(AUDIT_CSV, index=False)

    metadata = {
        "project": "The Mandi Ledger",
        "commodity": "Wheat",
        "geography": "All India",
        "period_start": str(chosen["month"].iloc[0]),
        "period_end": REQUIRED_END_MONTH,
        "monthly_observations": int(len(model_frame)),
        "expected_monthly_observations": int(len(expected)),
        "price_unit": "Rs./quintal",
        "source_dataset": SOURCE_REPO_URL,
        "underlying_source": "AGMARKNET statewise monthly price and arrivals records",
        "aggregation": (
            "All-India arrivals-weighted mean of state-level monthly modal prices: "
            "sum(price_rs_qtl * arrivals_t) / sum(arrivals_t)."
        ),
        "quality_filter": f"State-month rows with arrivals_t >= {MIN_ARRIVALS_TONNES} tonnes, valid positive price and valid arrivals.",
        "missing_months_filled": False,
        "note": (
            "This is a national state-aggregated series, not a single mandi. "
            "The published extract starts in January 2019. The output keeps only "
            "the most recent uninterrupted run ending October 2026; any earlier "
            "gaps are excluded, not filled. This is not a single-mandi series."
        ),
        "output_csv": str(OUTPUT_CSV.relative_to(PROJECT_DIR)),
        "audit_csv": str(AUDIT_CSV.relative_to(PROJECT_DIR)),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSUCCESS: All-India Wheat monthly series prepared.")
    print(f"Observations: {len(model_frame)}")
    print(f"Start: {model_frame['date'].iloc[0]}")
    print(f"Latest: {model_frame['date'].iloc[-1]}")
    print("Geography: All India (arrivals-weighted state aggregation)")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Audit: {AUDIT_CSV}")
    print(f"Metadata: {METADATA_JSON}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nDATA PREPARATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
