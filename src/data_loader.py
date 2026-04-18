import pandas as pd


def load_climate_data(filepath):
    """Load climate dataset and parse date column."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_climate_data(df):
    """Clean climate dataset and add derived features."""
    df = df.copy()
    df = df.drop_duplicates(subset=["date"])
    df = df.dropna(subset=["date"])
    df = df.set_index("date")

    numeric_cols = [
        "temperature_c",
        "rainfall_mm",
        "humidity_pct",
        "co2_ppm",
        "sea_level_mm",
        "wind_speed_kmh",
        "aqi_index",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="time").bfill().ffill()

    df = df.reset_index()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["season"] = df["month"].apply(lambda x: "Winter" if x in [12, 1, 2] else "Spring" if x in [3, 4, 5] else "Summer" if x in [6, 7, 8] else "Autumn")

    df["temp_rolling_12m"] = df["temperature_c"].rolling(window=12, min_periods=1).mean()
    df["rainfall_rolling_12m"] = df["rainfall_mm"].rolling(window=12, min_periods=1).mean()

    return df


def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned climate data to {output_path}")
