import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_climate_data, clean_climate_data
from src.visualize import create_interactive_trend_chart, create_seasonal_boxplot, create_correlation_chart
from src.forecast import linear_trend_forecast, sarimax_forecast, calculate_forecast_metrics
from src.anomaly import detect_anomalies_zscore

DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(
    page_title="Climate Trend Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding: 1rem 2rem 2rem 2rem;}
    .stMetric {border: 1px solid #eee; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Climate Trend Analyzer")
st.markdown("#### Professional climate analytics dashboard with trend detection, anomaly reporting, and forecast modeling.")

@st.cache_data
def load_clean_data(filepath):
    df = load_climate_data(filepath)
    return clean_climate_data(df)

@st.cache_data
def load_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return clean_climate_data(df)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

raw_file = DATA_DIR / "sample_climate_data.csv"
uploaded_file = st.sidebar.file_uploader("Upload your own climate dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)
        st.sidebar.success("Uploaded data loaded successfully.")
    except Exception as exc:
        st.sidebar.error(f"Upload failed: {exc}")
        df = load_clean_data(raw_file)
else:
    df = load_clean_data(raw_file)

if df.empty:
    st.error("No climate data available. Please upload a valid CSV or regenerate the sample dataset.")
    st.stop()

available_vars = [
    col for col in [
        "temperature_c",
        "rainfall_mm",
        "humidity_pct",
        "co2_ppm",
        "sea_level_mm",
        "wind_speed_kmh",
        "aqi_index",
    ]
    if col in df.columns
]

location_column = None
for candidate in ["location", "region", "area"]:
    if candidate in df.columns:
        location_column = candidate
        break

st.sidebar.header("Dashboard Controls")
selected_variable = st.sidebar.selectbox(
    "Select variable for trend analysis",
    available_vars,
    index=0,
)

if location_column:
    selected_location = st.sidebar.selectbox(
        "Select region",
        sorted(df[location_column].unique()),
        index=0,
    )
    df = df[df[location_column] == selected_location]
else:
    selected_location = "Global"

if df.empty:
    st.error("No data matches the selected region.")
    st.stop()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(df["date"].min().date(), df["date"].max().date()),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date(),
)
if isinstance(date_range, (tuple, list)):
    min_date, max_date = date_range
else:
    min_date = max_date = date_range

show_anomaly_table = st.sidebar.checkbox("Show anomaly table", value=True)
show_data_table = st.sidebar.checkbox("Show raw data table", value=False)
model_type = st.sidebar.selectbox(
    "Forecast model",
    ["Both", "Linear trend", "SARIMAX"],
    index=0,
)
evaluation_period = st.sidebar.slider(
    "Evaluation window (months)",
    min_value=6,
    max_value=24,
    value=12,
    step=6,
)

start_date = pd.to_datetime(min_date)
end_date = pd.to_datetime(max_date)
filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

if filtered_df.empty:
    st.warning("No data is available for the selected date range. Please choose a wider range.")
    st.stop()

tabs = st.tabs(["Overview", "Trends", "Anomalies", "Forecast", "Data", "About"])

with tabs[0]:
    st.subheader("Climate KPI dashboard")
    st.markdown(
        "This overview shows the latest climate indicators, trend direction, and dataset coverage for your selected region and date range."
    )

    latest_date = filtered_df["date"].max().date()
    first_date = filtered_df["date"].min().date()
    avg_temp = filtered_df["temperature_c"].mean() if "temperature_c" in filtered_df else None
    avg_rain = filtered_df["rainfall_mm"].mean() if "rainfall_mm" in filtered_df else None
    avg_humidity = filtered_df["humidity_pct"].mean() if "humidity_pct" in filtered_df else None
    latest_co2 = filtered_df["co2_ppm"].iloc[-1] if "co2_ppm" in filtered_df else None
    sea_level_change = (
        filtered_df["sea_level_mm"].iloc[-1] - filtered_df["sea_level_mm"].iloc[0]
        if "sea_level_mm" in filtered_df
        else None
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Date range", f"{first_date} to {latest_date}")
    metric2.metric("Region", selected_location)
    metric3.metric("Latest update", f"{latest_date}")
    metric4.metric("Selected variable", selected_variable.replace("_", " ").title())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Avg temperature (°C)", f"{avg_temp:.2f}" if avg_temp is not None else "N/A")
    kpi2.metric("Avg rainfall (mm)", f"{avg_rain:.1f}" if avg_rain is not None else "N/A")
    kpi3.metric("Avg humidity (%)", f"{avg_humidity:.1f}" if avg_humidity is not None else "N/A")
    kpi4.metric("Latest CO₂ (ppm)", f"{latest_co2:.1f}" if latest_co2 is not None else "N/A")

    if sea_level_change is not None:
        st.info(f"Sea level change over the selected period: {sea_level_change:.1f} mm")

    if "wind_speed_kmh" in filtered_df.columns:
        avg_wind = filtered_df["wind_speed_kmh"].mean()
        st.metric("Avg wind speed (km/h)", f"{avg_wind:.1f}")

    if "aqi_index" in filtered_df.columns:
        avg_aqi = filtered_df["aqi_index"].mean()
        st.metric("Avg AQI", f"{avg_aqi:.1f}")

    st.markdown("---")
    st.markdown(
        "### Professional climate analysis narrative\n"
        "Use this dashboard to compare climate indicators, detect anomalies, and forecast future values for better environmental decision-making. "
        "The dataset can be updated with your own CSV file for custom analysis."
    )

with tabs[1]:
    st.subheader("Trend analysis")
    st.markdown("Explore how climate variables change over time and compare seasonal patterns with an interactive correlation view.")
    st.plotly_chart(
        create_interactive_trend_chart(filtered_df, "date", selected_variable, f"{selected_variable.replace('_', ' ').title()} Trend"),
        use_container_width=True,
    )

    season_col1, season_col2 = st.columns(2)
    if selected_variable in ["temperature_c", "rainfall_mm"]:
        if selected_variable == "temperature_c":
            season_col1.plotly_chart(
                create_seasonal_boxplot(filtered_df, "temperature_c", "Temperature by Season"),
                use_container_width=True,
            )
            if "rainfall_mm" in filtered_df.columns:
                season_col2.plotly_chart(
                    create_seasonal_boxplot(filtered_df, "rainfall_mm", "Rainfall by Season"),
                    use_container_width=True,
                )
        else:
            season_col1.plotly_chart(
                create_seasonal_boxplot(filtered_df, selected_variable, f"{selected_variable.replace('_', ' ').title()} by Season"),
                use_container_width=True,
            )
            if "temperature_c" in filtered_df.columns:
                season_col2.plotly_chart(
                    create_seasonal_boxplot(filtered_df, "temperature_c", "Temperature by Season"),
                    use_container_width=True,
                )
    else:
        season_col1.plotly_chart(
            create_seasonal_boxplot(filtered_df, selected_variable, f"{selected_variable.replace('_', ' ').title()} by Season"),
            use_container_width=True,
        )
        if "temperature_c" in filtered_df.columns:
            season_col2.plotly_chart(
                create_seasonal_boxplot(filtered_df, "temperature_c", "Temperature by Season"),
                use_container_width=True,
            )

    correlation_vars = [col for col in available_vars if col in filtered_df.columns]
    if len(correlation_vars) >= 3:
        st.plotly_chart(
            create_correlation_chart(filtered_df, correlation_vars, "Climate Variable Correlations"),
            use_container_width=True,
        )

with tabs[2]:
    st.subheader("Anomaly detection")
    st.markdown("Detect extreme climate events by seeing which values are outside normal seasonal behavior.")
    threshold = st.slider("Anomaly sensitivity (z-score)", min_value=1.5, max_value=4.0, value=2.5, step=0.1)

    anomalies = detect_anomalies_zscore(filtered_df, selected_variable, threshold=threshold)

    anomaly_col1, anomaly_col2 = st.columns(2)
    anomaly_col1.metric("Detected anomalies", len(anomalies))
    anomaly_col2.metric("Monitored variable", selected_variable.replace("_", " ").title())

    if show_anomaly_table:
        if not anomalies.empty:
            st.write(f"#### Recent anomalies for {selected_variable.replace('_', ' ').title()}")
            st.dataframe(anomalies[["date", selected_variable, "z_score"]].sort_values("date", ascending=False).head(15))
        else:
            st.info("No anomalies found for the selected variable and date range.")
    st.markdown(
        "- Values outside the z-score threshold are flagged as anomalies.\n"
        "- Use a higher threshold for stricter anomaly detection and a lower threshold for sensitivity."
    )

with tabs[3]:
    st.subheader("Forecast and model evaluation")
    st.markdown("Use the forecast section to compare forecasting models and evaluate predictions against recent historical observations.")
    forecast_horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=24, value=12)

    forecast_results = []
    if model_type in ["Both", "Linear trend"]:
        linear_forecast, _ = linear_trend_forecast(filtered_df, selected_variable, periods=forecast_horizon)
        forecast_results.append(("Linear trend", linear_forecast))
    if model_type in ["Both", "SARIMAX"]:
        sarimax_forecast_df, _ = sarimax_forecast(filtered_df, selected_variable, periods=forecast_horizon)
        forecast_results.append(("SARIMAX", sarimax_forecast_df))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["date"],
            y=filtered_df[selected_variable],
            mode="lines",
            name="Historical",
            line=dict(color="#2C7BE5"),
        )
    )
    for model_name, forecast_df in forecast_results:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df[f"forecast_{selected_variable}"],
                mode="lines+markers",
                name=model_name,
            )
        )
    fig.update_layout(
        title=f"Forecast for {selected_variable.replace('_', ' ').title()}",
        xaxis_title="Date",
        yaxis_title=selected_variable.replace("_", " ").title(),
    )
    st.plotly_chart(fig, use_container_width=True)

    if forecast_results:
        next_values = []
        for model_name, forecast_df in forecast_results:
            next_val = forecast_df[f"forecast_{selected_variable}"].iloc[0]
            next_date = forecast_df["date"].iloc[0].date()
            next_values.append((model_name, next_date, next_val))

        predict_cols = st.columns(len(next_values))
        for idx, (model_name, next_date, next_val) in enumerate(next_values):
            predict_cols[idx].metric(
                f"Next month ({model_name})",
                f"{next_val:.2f}",
                help=f"For {next_date}",
            )

    forecast_table_col1, forecast_table_col2 = st.columns(2)
    for model_name, forecast_df in forecast_results:
        if model_name == "Linear trend":
            forecast_table_col1.write(f"#### {model_name} forecast")
            forecast_table_col1.table(forecast_df.head(5))
        else:
            forecast_table_col2.write(f"#### {model_name} forecast")
            forecast_table_col2.table(forecast_df.head(5))

    if len(filtered_df) >= evaluation_period + 12:
        test_df = filtered_df.iloc[-evaluation_period:]
        train_df = filtered_df.iloc[:-evaluation_period]
        st.markdown("---")
        st.write("### Model accuracy")

        eval_results = []
        if model_type in ["Both", "Linear trend"]:
            linear_eval, _ = linear_trend_forecast(train_df, selected_variable, periods=evaluation_period)
            linear_metrics = calculate_forecast_metrics(test_df[selected_variable].values, linear_eval[f"forecast_{selected_variable}"].values)
            eval_results.append(("Linear trend", linear_metrics))
        if model_type in ["Both", "SARIMAX"]:
            sarimax_eval, _ = sarimax_forecast(train_df, selected_variable, periods=evaluation_period)
            sarimax_metrics = calculate_forecast_metrics(test_df[selected_variable].values, sarimax_eval[f"forecast_{selected_variable}"].values)
            eval_results.append(("SARIMAX", sarimax_metrics))

        eval_table = pd.DataFrame(
            [
                {"Model": name, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"]}
                for name, metrics in eval_results
            ]
        )
        st.table(eval_table)
    else:
        st.info("Not enough history to compute model accuracy. Expand the selected date range to enable evaluation.")

    if forecast_results:
        combined_forecast = pd.concat([df for _, df in forecast_results], ignore_index=True)
        st.download_button(
            label="Download forecast data",
            data=convert_df_to_csv(combined_forecast),
            file_name="climate_forecast_export.csv",
            mime="text/csv",
        )

with tabs[4]:
    st.subheader("Data and exports")
    st.markdown("Download the cleaned dataset or summary statistics for reporting and portfolio documentation.")
    if show_data_table:
        st.dataframe(filtered_df)

    st.download_button(
        label="Download cleaned dataset",
        data=convert_df_to_csv(filtered_df),
        file_name="cleaned_climate_dataset.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download summary statistics",
        data=convert_df_to_csv(filtered_df.describe().round(2)),
        file_name="climate_summary_statistics.csv",
        mime="text/csv",
    )

with tabs[5]:
    st.subheader("About this dashboard")
    st.markdown(
        "This professional climate dashboard includes:\n"
        "- custom dataset upload\n"
        "- KPI metrics and region analysis\n"
        "- seasonal trend charts and correlation analysis\n"
        "- anomaly detection with adjustable z-score sensitivity\n"
        "- forecast comparison between model types\n"
        "- data export for reports and presentations"
    )
    st.markdown(
        "This project is ideal for student portfolios, interviews, and climate analytics prototypes. "
        "It demonstrates core skills in data cleaning, time-series analysis, forecasting, and interactive reporting."
    )
    st.write("---")
    st.write("### Deployment note")
    st.write(
        "Run `streamlit run app/streamlit_app.py` from the project root. "
        "This app can be deployed to Streamlit Community Cloud for a live website demo."
    )
