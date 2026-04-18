import plotly.express as px


def create_interactive_trend_chart(df, x_col, y_col, title):
    fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
    fig.update_layout(xaxis_title="Date", yaxis_title=y_col.replace("_", " ").title())
    return fig


def create_correlation_chart(df, cols, title):
    fig = px.imshow(df[cols].corr(), text_auto=True, color_continuous_scale="RdBu_r")
    fig.update_layout(title=title)
    return fig


def create_seasonal_boxplot(df, value_col, title):
    fig = px.box(df, x="season", y=value_col, color="season", title=title)
    fig.update_layout(xaxis_title="Season", yaxis_title=value_col.replace("_", " ").title())
    return fig
