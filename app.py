"""Optional Streamlit interface for the Crop Price Prediction Model."""
from pathlib import Path
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from crop_price_model import load_monthly_data, train_evaluate

st.set_page_config(page_title="The Mandi Ledger | Crop Price Lab", page_icon="🌾", layout="wide")
st.title("The Mandi Ledger")
st.caption("Crop Price Prediction Lab | Linear Regression baseline")
st.warning(
    "The bundled dataset is synthetic demonstration data. Its model scores and forecast are not real mandi-price evidence. "
    "Use a verified, single-crop/single-market series before drawing market conclusions."
)

base = Path(__file__).resolve().parent
upload = st.file_uploader("Upload a monthly CSV", type=["csv"])
source = upload if upload is not None else base / "demo_crop_price_data.csv"
if upload is None:
    st.info("Using the included synthetic demo dataset. Upload your cleaned CSV to evaluate real observations.")

try:
    df = load_monthly_data(source)
    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly observations", f"{len(df)}")
    c2.metric("Start month", df.date.min().strftime("%b %Y"))
    c3.metric("End month", df.date.max().strftime("%b %Y"))
    st.subheader("Observed monthly series")
    st.line_chart(df.set_index("date")["modal_price_rs_per_quintal"])

    if st.button("Train model and evaluate", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            provenance = "SYNTHETIC DEMONSTRATION DATA - NOT REAL MARKET EVIDENCE" if upload is None else "UPLOADED DATA - USER MUST VERIFY SOURCE AND COLLECTION METHOD"
            metrics = train_evaluate(df, Path(tmp), data_status=provenance)
            st.subheader("Holdout performance")
            m1, m2 = st.columns(2)
            for col, title in [(m1, "Linear Regression"), (m2, "Last-price baseline")]:
                values = metrics["linear_regression"] if title == "Linear Regression" else metrics["naive_last_price_baseline"]
                with col:
                    st.markdown(f"**{title}**")
                    st.metric("MAE (₹/quintal)", f"{values['MAE_Rs_per_quintal']:,.2f}")
                    st.metric("RMSE (₹/quintal)", f"{values['RMSE_Rs_per_quintal']:,.2f}")
                    st.metric("R²", f"{values['R2']:.3f}")
            f = metrics["next_month_forecast"]
            st.subheader(f"One-month-ahead estimate: {f['month']}")
            st.metric("Predicted monthly price (₹/quintal)", f"₹{f['predicted_modal_price_rs_per_quintal']:,.2f}")
            st.caption("This is a statistical demonstration, not financial or farming advice. Check data provenance and model errors.")
            for filename, title in [
                ("actual_vs_predicted.png", "Holdout predictions"),
                ("residuals.png", "Residual inspection"),
                ("price_trend.png", "Price history"),
            ]:
                st.markdown(f"**{title}**")
                st.image(str(Path(tmp) / filename), use_container_width=True)
            st.download_button(
                "Download holdout predictions CSV",
                data=(Path(tmp) / "test_predictions.csv").read_bytes(),
                file_name="test_predictions.csv",
                mime="text/csv",
            )
except Exception as exc:
    st.error(str(exc))

st.divider()
st.markdown("**Input format:** `date, modal_price_rs_per_quintal` — one continuous monthly series for one crop and one market. Multiple daily AGMARKNET rows are aggregated to monthly mean daily modal price by the training module.")
