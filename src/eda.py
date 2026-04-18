import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_series(df, x_col, y_col, title, output_path=None):
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x=x_col, y=y_col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_col.replace("_", " ").title())
    plt.grid(alpha=0.3)
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_seasonal_pattern(df, value_col, title, output_path=None):
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x="month", y=value_col, hue="year", estimator="mean", palette="tab10", legend=False)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(value_col.replace("_", " ").title())
    plt.grid(alpha=0.3)
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_correlation_heatmap(df, cols, title, output_path=None):
    plt.figure(figsize=(10, 8))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(title)
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
    plt.show()


def save_summary_tables(df, output_folder):
    summary = df.describe().round(2)
    summary_file = output_folder / "climate_summary_statistics.csv"
    summary.to_csv(summary_file)
    print(f"Saved summary statistics to {summary_file}")
