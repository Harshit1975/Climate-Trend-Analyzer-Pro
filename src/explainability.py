import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go


def get_linear_feature_importance(df, target_col):
    """Calculate feature importance for linear regression model."""
    df = df.copy()
    df["time_index"] = np.arange(len(df))
    
    # Only select numeric columns
    feature_cols = [col for col in df.columns if col not in ["date", "location", target_col] and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    if len(feature_cols) == 0:
        return pd.DataFrame({"Feature": [target_col], "Importance": [100.0]}), None
    
    X = df[feature_cols]
    y = df[target_col]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Normalize coefficients for importance
    abs_coefs = np.abs(model.coef_)
    importance = (abs_coefs / abs_coefs.sum()) * 100
    
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": np.round(importance, 2)
    }).sort_values("Importance", ascending=True)
    
    return importance_df, model


def get_tree_feature_importance(df, target_col, n_estimators=100):
    """Calculate feature importance using Random Forest."""
    df = df.copy()
    
    # Only select numeric columns
    feature_cols = [col for col in df.columns if col not in ["date", "location", target_col] and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    if len(feature_cols) == 0:
        return pd.DataFrame({"Feature": [target_col], "Importance": [100.0]}), None
    
    X = df[feature_cols]
    y = df[target_col]
    
    if len(X) < 10:
        return pd.DataFrame({"Feature": feature_cols, "Importance": [0] * len(feature_cols)}), None
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=5)
    model.fit(X, y)
    
    importance = (model.feature_importances_ / model.feature_importances_.sum()) * 100
    
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": np.round(importance, 2)
    }).sort_values("Importance", ascending=True)
    
    return importance_df, model


def create_feature_importance_chart(importance_df, title="Feature Importance"):
    """Create an interactive feature importance chart."""
    fig = go.Figure(data=[
        go.Bar(
            y=importance_df["Feature"],
            x=importance_df["Importance"],
            orientation="h",
            marker=dict(color="#3498db")
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def calculate_rolling_correlation(df, target_col, window=12):
    """Calculate rolling correlation with other variables."""
    df = df.copy()
    numeric_cols = [col for col in df.columns if col not in ["date", "location"] and df[col].dtype in [np.float64, np.int64]]
    
    rolling_corr = {}
    for col in numeric_cols:
        if col != target_col:
            rolling_corr[col] = df[target_col].rolling(window).corr(df[col])
    
    return pd.DataFrame(rolling_corr, index=df["date"])
