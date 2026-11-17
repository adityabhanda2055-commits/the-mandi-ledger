
"""The Mandi Ledger — Agri-Market Intelligence Dashboard."""

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from crop_price_model import load_monthly_data, train_evaluate


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="The Mandi Ledger | Agri-Market Intelligence",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 The Mandi Ledger")
st.subheader("Agri-Market Intelligence & Crop Price Forecasting")

st.markdown(
    """
    Analyse historical mandi prices, compare machine-learning models,
    evaluate prediction performance and explore projected crop prices
    through December 2027.
    """
)
# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def find_column(dataframe, hints):
    """Find a column whose name contains one of the supplied hints."""
    for hint in hints:
        for column in dataframe.columns:
            if hint in str(column).lower():
                return column
    return None


def show_download(filename, label, directory, mime="text/csv"):
    """Display a download button when an output file exists."""
    file_path = directory / filename

    if file_path.exists():
        st.download_button(
            label=label,
            data=file_path.read_bytes(),
            file_name=filename,
            mime=mime,
            key=f"download_{filename}",
        )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("Project Information")

    st.markdown("**Project:** The Mandi Ledger")
    st.markdown("**Domain:** Agricultural Analytics")
    st.markdown("**Task:** Crop Price Prediction")
    st.markdown("**Forecast horizon:** December 2027")

    st.divider()

    st.markdown("### Machine Learning Models")
    st.markdown(
        """
        - Linear Regression
        - Ridge Regression
        - Random Forest
        - HistGradientBoosting
        """
    )

    st.divider()

    st.markdown("### Evaluation Metrics")
    st.markdown(
        """
        - MAE
        - RMSE
        - R² Score
        """
    )


# ---------------------------------------------------------
# DATA INPUT
# ---------------------------------------------------------

st.header("1. Data Foundation")

uploaded_file = st.file_uploader(
    "Upload your crop-price CSV file",
    type=["csv"],
    help="Upload a continuous price history for one crop and one market.",
)


base_directory = Path(__file__).resolve().parent

real_data_path = (
    base_directory / "data" / "real_mandi_monthly.csv"
)

demo_data_path = (
    base_directory / "demo_crop_price_data.csv"
)

if uploaded_file is not None:
    data_source = uploaded_file
    data_status = "USER-UPLOADED DATA - SOURCE TO BE VERIFIED"

elif real_data_path.exists():
    data_source = real_data_path
    data_status = (
        "REAL AGMARKNET-DERIVED DATA - "
        "ALL-INDIA ARRIVALS-WEIGHTED WHEAT PRICE"
    )
    st.success("Using real All-India Wheat price data.")

else:
    data_source = demo_data_path
    data_status = "SYNTHETIC DEMONSTRATION DATA"

    st.warning("Using synthetic demonstration data.")


# ---------------------------------------------------------
# LOAD AND DISPLAY DATA
# ---------------------------------------------------------

try:
    df = load_monthly_data(data_source)

    st.success("Dataset loaded and validated successfully.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Monthly Observations", len(df))

    col2.metric(
        "Starting Month",
        df["date"].min().strftime("%b %Y"),
    )

    col3.metric(
        "Latest Month",
        df["date"].max().strftime("%b %Y"),
    )

    col4.metric(
        "Average Price",
        f"₹{df['modal_price_rs_per_quintal'].mean():,.2f}",
    )

    st.subheader("Historical Crop Price Trend")

    historical_chart = df.set_index("date")[
        "modal_price_rs_per_quintal"
    ]

    st.line_chart(historical_chart)

    with st.expander("View uploaded or demonstration data"):
        st.dataframe(df, use_container_width=True)

    # -----------------------------------------------------
    # MODEL TRAINING
    # -----------------------------------------------------

    st.divider()
    st.header("2. Machine Learning & Model Evaluation")

    st.markdown(
        """
        Train multiple models using a chronological train-test split.
        The comparison helps identify which model performs best on
        the available holdout observations.
        """
    )

    if st.button(
        "🚀 Train Models & Generate Forecast",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Training models, evaluating predictions and generating forecasts..."
        ):

            with tempfile.TemporaryDirectory() as temporary_directory:

                output_directory = Path(temporary_directory)

                try:
                    metrics = train_evaluate(
                        df,
                        output_directory,
                        data_status=data_status,
                    )

                    st.success(
                        "Model training and forecast generation completed."
                    )

                    # -------------------------------------
                    # MODEL COMPARISON
                    # -------------------------------------

                    st.subheader("Model Performance Comparison")

                    comparison_path = (
                        output_directory / "model_comparison.csv"
                    )

                    if comparison_path.exists():

                        comparison = pd.read_csv(comparison_path)

                        st.dataframe(
                            comparison,
                            use_container_width=True,
                            hide_index=True,
                        )

                        mae_column = find_column(
                            comparison,
                            ["mae"],
                        )

                        model_column = find_column(
                            comparison,
                            ["model", "algorithm", "method"],
                        )

                        rmse_column = find_column(
                            comparison,
                            ["rmse"],
                        )

                        r2_column = find_column(
                            comparison,
                            ["r2"],
                        )

                        if mae_column and model_column:

                            comparison[mae_column] = pd.to_numeric(
                                comparison[mae_column],
                                errors="coerce",
                            )

                            candidates = comparison.dropna(
                                subset=[mae_column]
                            ).copy()

                            # Exclude baseline rows when selecting
                            # the best machine-learning model.
                            baseline_mask = candidates[
                                model_column
                            ].astype(str).str.contains(
                                "baseline|naive",
                                case=False,
                                regex=True,
                            )

                            machine_learning_candidates = candidates[
                                ~baseline_mask
                            ]

                            if not machine_learning_candidates.empty:
                                best_row = machine_learning_candidates.loc[
                                    machine_learning_candidates[
                                        mae_column
                                    ].idxmin()
                                ]

                                st.subheader("Best Performing ML Model")

                                best_col1, best_col2, best_col3 = (
                                    st.columns(3)
                                )

                                best_col1.metric(
                                    "Selected Model",
                                    str(best_row[model_column]),
                                )

                                best_col2.metric(
                                    "MAE",
                                    f"{best_row[mae_column]:,.2f}",
                                )

                                if (
                                    rmse_column
                                    and pd.notna(best_row[rmse_column])
                                ):
                                    best_col3.metric(
                                        "RMSE",
                                        f"{float(best_row[rmse_column]):,.2f}",
                                    )

                                if (
                                    r2_column
                                    and pd.notna(best_row[r2_column])
                                ):
                                    st.metric(
                                        "Best Model R² Score",
                                        f"{float(best_row[r2_column]):.3f}",
                                    )

                    else:
                        st.warning(
                            "Model comparison CSV was not generated. "
                            "Check the training module."
                        )

                    # -------------------------------------
                    # NEXT-MONTH FORECAST
                    # -------------------------------------

                    st.divider()
                    st.subheader("Next-Month Price Estimate")

                    next_forecast = metrics.get(
                        "next_month_forecast"
                    )

                    if next_forecast:

                        forecast_col1, forecast_col2 = st.columns(2)

                        forecast_col1.metric(
                            "Forecast Month",
                            str(next_forecast["month"]),
                        )

                        forecast_col2.metric(
                            "Predicted Price",
                            (
                                "₹"
                                f"{next_forecast['predicted_modal_price_rs_per_quintal']:,.2f}"
                            ),
                            "Per quintal",
                        )

                    # -------------------------------------
                    # FORECAST THROUGH 2027
                    # -------------------------------------

                    st.divider()
                    st.header("3. Crop Price Forecast Through 2027")

                    forecast_csv = (
                        output_directory / "forecast_to_2027.csv"
                    )

                    if forecast_csv.exists():

                        forecast_data = pd.read_csv(forecast_csv)

                        st.success(
                            "Forecast dataset generated successfully."
                        )

                        st.dataframe(
                            forecast_data,
                            use_container_width=True,
                            hide_index=True,
                        )

                        date_column = find_column(
                            forecast_data,
                            ["date", "month"],
                        )

                        predicted_column = find_column(
                            forecast_data,
                            ["predicted", "forecast"],
                        )

                        if date_column and predicted_column:

                            chart_data = forecast_data[
                                [date_column, predicted_column]
                            ].copy()

                            chart_data[date_column] = pd.to_datetime(
                                chart_data[date_column],
                                errors="coerce",
                            )

                            chart_data[predicted_column] = pd.to_numeric(
                                chart_data[predicted_column],
                                errors="coerce",
                            )

                            chart_data = chart_data.dropna()

                            if not chart_data.empty:

                                st.subheader(
                                    "Projected Monthly Crop Prices"
                                )

                                chart_data = chart_data.sort_values(
                                    date_column
                                )

                                st.line_chart(
                                    chart_data.set_index(date_column)[
                                        predicted_column
                                    ]
                                )

                    else:
                        st.warning(
                            "The 2027 forecast CSV was not found."
                        )

                    # -------------------------------------
                    # MODEL VISUALIZATIONS
                    # -------------------------------------

                    st.divider()
                    st.header("4. Model Visualizations")

                    visualizations = [
                        (
                            "actual_vs_predicted.png",
                            "Actual vs Predicted Prices",
                        ),
                        (
                            "residuals.png",
                            "Residual Error Analysis",
                        ),
                        (
                            "price_trend.png",
                            "Historical Price Trend",
                        ),
                        (
                            "forecast_to_2027.png",
                            "Forecast Through December 2027",
                        ),
                    ]

                    for filename, title in visualizations:

                        image_path = output_directory / filename

                        if image_path.exists():
                            st.subheader(title)
                            st.image(
                                str(image_path),
                                use_container_width=True,
                            )

                    # -------------------------------------
                    # DOWNLOADABLE REPORTS
                    # -------------------------------------

                    st.divider()
                    st.header("5. Download Project Results")

                    download_col1, download_col2 = st.columns(2)

                    with download_col1:

                        show_download(
                            "model_comparison.csv",
                            "Download Model Comparison",
                            output_directory,
                        )

                        show_download(
                            "test_predictions.csv",
                            "Download Test Predictions",
                            output_directory,
                        )

                    with download_col2:

                        show_download(
                            "forecast_to_2027.csv",
                            "Download 2027 Forecast",
                            output_directory,
                        )

                        show_download(
                            "metrics.json",
                            "Download Evaluation Metrics",
                            output_directory,
                            mime="application/json",
                        )

                    st.caption(
                        "Forecasts are statistical estimates based on "
                        "historical inputs. They are not guaranteed prices "
                        "or financial, farming or trading advice. "
                        "Forecast uncertainty generally increases with horizon."
                    )

                except Exception as training_error:

                    st.error(
                        "The training process could not be completed."
                    )

                    with st.expander("View technical error"):
                        st.exception(training_error)


except Exception as data_error:

    st.error(
        "The dataset could not be loaded. Check its columns, date format "
        "and whether it contains enough continuous monthly observations."
    )

    with st.expander("View technical error"):
        st.exception(data_error)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "THE MANDI LEDGER | Agricultural Data Analytics | "
    "Machine Learning Demonstration Project"
)

st.caption(
    "Input format: date and modal_price_rs_per_quintal, "
    "or supported AGMARKNET-style date and modal-price columns. "
    "Use one continuous crop-market series."
)
