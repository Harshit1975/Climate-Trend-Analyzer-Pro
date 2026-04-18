import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm


def prepare_forecast_features(df, target_col):
    df = df.copy()
    df["time_index"] = np.arange(len(df))
    X = df[["time_index"]]
    y = df[target_col]
    return X, y


def linear_trend_forecast(df, target_col, periods=12, confidence=0.95):
    X, y = prepare_forecast_features(df, target_col)
    model = LinearRegression()
    model.fit(X, y)

    future_index = pd.DataFrame({"time_index": np.arange(len(df), len(df) + periods)})
    forecast = model.predict(future_index)
    
    # Calculate prediction intervals
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    z_score = 1.96 if confidence == 0.95 else 1.64 if confidence == 0.90 else 2.58
    margin = z_score * std_error
    
    lower_bound = forecast - margin
    upper_bound = forecast + margin

    future_dates = pd.date_range(start=df["date"].max() + pd.DateOffset(months=1), periods=periods, freq="MS")
    forecast_df = pd.DataFrame({
        "date": future_dates, 
        f"forecast_{target_col}": np.round(forecast, 2),
        f"lower_bound_{target_col}": np.round(lower_bound, 2),
        f"upper_bound_{target_col}": np.round(upper_bound, 2)
    })
    return forecast_df, model


def sarimax_forecast(df, target_col, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), periods=12):
    ts = df.set_index("date")[target_col]
    ts = ts.asfreq("MS")
    model = sm.tsa.SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()[["mean", "mean_ci_lower", "mean_ci_upper"]].reset_index()
    forecast_df.columns = ["date", f"forecast_{target_col}", f"lower_bound_{target_col}", f"upper_bound_{target_col}"]
    forecast_df[f"forecast_{target_col}"] = forecast_df[f"forecast_{target_col}"].round(2)
    forecast_df[f"lower_bound_{target_col}"] = forecast_df[f"lower_bound_{target_col}"].round(2)
    forecast_df[f"upper_bound_{target_col}"] = forecast_df[f"upper_bound_{target_col}"].round(2)
    return forecast_df, results


def calculate_forecast_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}
