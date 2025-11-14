import pandas as pd
import plotly.express as px
from typing import Optional

def plot_loudness_trend(
    year_data: pd.DataFrame,
    year_col: str = "year",
    loudness_col: str = "loudness",
    title: Optional[str] = None,
    markers: bool = True,
    template: str = "plotly_white",
    save_path: Optional[str] = None,
    show: bool = True,
) -> "plotly.graph_objs._figure.Figure":
    """
    Plot loudness trend over years/decades using plotly.express.line.

    - year_data: DataFrame with one row per year/decade and a mean 'loudness' column.
    - year_col: name of the column containing the year/decade values (default 'year').
    - loudness_col: name of the loudness column (default 'loudness').
    - save_path: optional path to save the interactive plot (HTML).
    - show: whether to call fig.show().
    Returns the plotly Figure.
    """
    if year_col not in year_data.columns:
        raise ValueError(f"year_col '{year_col}' not found in DataFrame.")
    if loudness_col not in year_data.columns:
        raise ValueError(f"loudness_col '{loudness_col}' not found in DataFrame.")

    df = year_data.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col])
    df = df.sort_values(by=year_col)

    if title is None:
        title = "Trend of loudness over decades"

    fig = px.line(
        df,
        x=year_col,
        y=loudness_col,
        title=title,
        markers=markers,
        template=template,
    )
    fig.update_layout(xaxis_title=year_col, yaxis_title="Loudness (mean)")

    if save_path:
        fig.write_html(save_path)

    if show:
        fig.show()

    return fig