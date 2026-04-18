import numpy as np


def detect_anomalies_zscore(df, column, threshold=2.5):
    """Detect anomalies using z-score on a numeric column."""
    values = df[column].values
    mean = np.mean(values)
    std = np.std(values)
    z_scores = (values - mean) / std
    anomalies = df[np.abs(z_scores) > threshold].copy()
    anomalies["z_score"] = z_scores[np.abs(z_scores) > threshold]
    return anomalies


def detect_anomalies_iqr(df, column):
    """Detect anomalies using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    anomalies = df[(df[column] < lower) | (df[column] > upper)].copy()
    anomalies["lower_bound"] = lower
    anomalies["upper_bound"] = upper
    return anomalies
