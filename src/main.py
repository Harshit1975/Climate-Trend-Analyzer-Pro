from pathlib import Path

from src.data_loader import load_climate_data, clean_climate_data, save_cleaned_data
from src.eda import plot_time_series, plot_seasonal_pattern, plot_correlation_heatmap, save_summary_tables
from src.anomaly import detect_anomalies_zscore, detect_anomalies_iqr
from src.forecast import linear_trend_forecast, sarimax_forecast


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_analysis():
    raw_file = DATA_DIR / "sample_climate_data.csv"
    cleaned_file = OUTPUT_DIR / "cleaned_climate_data.csv"

    df = load_climate_data(raw_file)
    df = clean_climate_data(df)
    save_cleaned_data(df, cleaned_file)

    plot_time_series(df, "date", "temperature_c", "Monthly Average Temperature Over Time", OUTPUT_DIR / "temperature_trend.png")
    plot_time_series(df, "date", "rainfall_mm", "Monthly Rainfall Trend", OUTPUT_DIR / "rainfall_trend.png")
    plot_time_series(df, "date", "co2_ppm", "Monthly CO2 Concentration Trend", OUTPUT_DIR / "co2_trend.png")

    plot_seasonal_pattern(df, "temperature_c", "Seasonal Temperature Pattern", OUTPUT_DIR / "temperature_seasonality.png")
    plot_seasonal_pattern(df, "rainfall_mm", "Seasonal Rainfall Pattern", OUTPUT_DIR / "rainfall_seasonality.png")

    plot_correlation_heatmap(df, ["temperature_c", "rainfall_mm", "humidity_pct", "co2_ppm", "sea_level_mm"], "Climate Variable Correlations", OUTPUT_DIR / "correlation_heatmap.png")
    save_summary_tables(df, OUTPUT_DIR)

    anomalies_temp = detect_anomalies_zscore(df, "temperature_c")
    anomalies_rain = detect_anomalies_zscore(df, "rainfall_mm")
    anomalies_temp.to_csv(OUTPUT_DIR / "temperature_anomalies.csv", index=False)
    anomalies_rain.to_csv(OUTPUT_DIR / "rainfall_anomalies.csv", index=False)
    print(f"Saved anomaly reports to {OUTPUT_DIR}")

    forecast_temp, lr_model = linear_trend_forecast(df, "temperature_c", periods=12)
    forecast_temp.to_csv(OUTPUT_DIR / "temperature_forecast_linear.csv", index=False)
    print("Saved temperature forecast with linear trend.")

    try:
        forecast_temp_sarimax, sarimax_model = sarimax_forecast(df, "temperature_c", periods=12)
        forecast_temp_sarimax.to_csv(OUTPUT_DIR / "temperature_forecast_sarimax.csv", index=False)
        print("Saved SARIMAX temperature forecast.")
    except Exception as err:
        print(f"SARIMAX forecasting failed: {err}")

    print("Analysis complete. Check outputs/ for charts and CSV reports.")


if __name__ == "__main__":
    run_analysis()
