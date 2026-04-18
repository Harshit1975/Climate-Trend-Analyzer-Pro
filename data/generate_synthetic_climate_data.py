import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent

def generate_synthetic_climate_data(start_year=2000, end_year=2024):
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="MS")
    n = len(dates)

    base_temp = 14.0
    seasonal_temp = 10 * np.sin(2 * np.pi * (dates.month - 1) / 12)
    trend_temp = 0.03 * (dates.year - start_year)
    noise_temp = np.random.normal(scale=0.8, size=n)
    temperature = base_temp + seasonal_temp + trend_temp + noise_temp

    month_factor = np.where(dates.month.isin([6, 7, 8, 9]), 120, 60)
    trend_rain = 0.2 * (dates.year - start_year)
    noise_rain = np.random.normal(scale=20, size=n)
    rainfall = np.clip(month_factor + trend_rain + noise_rain, 0, None)

    humidity = 55 + 20 * np.cos(2 * np.pi * (dates.month - 1) / 12) + np.random.normal(scale=5, size=n)
    humidity = np.clip(humidity, 20, 100)

    base_co2 = 370
    trend_co2 = 2.2 * (dates.year - start_year)
    seasonal_co2 = 4 * np.sin(2 * np.pi * (dates.month - 1) / 12)
    noise_co2 = np.random.normal(scale=0.5, size=n)
    co2 = base_co2 + trend_co2 + seasonal_co2 + noise_co2

    base_sea_level = 0
    trend_sea = 3.2 * (dates.year - start_year) + 0.02 * (dates.year - start_year) ** 2
    noise_sea = np.random.normal(scale=2.0, size=n)
    sea_level = base_sea_level + trend_sea + noise_sea

    wind_speed = np.clip(10 + 3 * np.sin(2 * np.pi * (dates.month - 1) / 12) + np.random.normal(scale=2.0, size=n), 1, 30)
    aqi = np.clip(60 + 1.0 * (dates.year - start_year) + 6 * np.cos(2 * np.pi * (dates.month - 1) / 12) + np.random.normal(scale=4.0, size=n), 20, 180)

    data = pd.DataFrame({
        "date": dates,
        "location": "Global",
        "temperature_c": np.round(temperature, 2),
        "rainfall_mm": np.round(rainfall, 1),
        "humidity_pct": np.round(humidity, 1),
        "co2_ppm": np.round(co2, 2),
        "sea_level_mm": np.round(sea_level, 2),
        "wind_speed_kmh": np.round(wind_speed, 1),
        "aqi_index": np.round(aqi, 1),
    })

    anomaly_indices = np.random.choice(n, size=max(1, n // 18), replace=False)
    data.loc[anomaly_indices, "temperature_c"] += np.random.choice([2.5, 3.5, -2.0], size=len(anomaly_indices))
    data.loc[anomaly_indices, "rainfall_mm"] += np.random.choice([40, -30, 50], size=len(anomaly_indices))

    return data


def main():
    df = generate_synthetic_climate_data(start_year=2000, end_year=2024)
    output_file = OUTPUT_PATH / "sample_climate_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Generated synthetic climate dataset at: {output_file}")

if __name__ == "__main__":
    main()
