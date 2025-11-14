import typing
import pandas as pd
import plotly.express as px

def plot_sound_feature_trends(
    year_data: pd.DataFrame,
    year_col: str = "year",
    sound_features: typing.Optional[typing.List[str]] = None,
    title: typing.Optional[str] = None,
    markers: bool = True,
    template: str = "plotly_white",
    save_path: typing.Optional[str] = None,
    show: bool = True,
) -> "plotly.graph_objs._figure.Figure":
    """
    Plot trends of multiple sound features over years/decades using plotly.express.line.

    - year_data: DataFrame indexed/grouped by year (or decade) with mean feature values per year.
    - year_col: column name containing the year/decade values (default 'year').
    - sound_features: list of column names to plot; default includes common Spotify features.
    - title: optional plot title.
    - markers: whether to show markers on the line.
    - save_path: if provided, save figure to this path as HTML.
    - show: whether to call fig.show().

    Returns the plotly Figure.
    """
    if sound_features is None:
        sound_features = [
            "acousticness",
            "danceability",
            "energy",
            "instrumentalness",
            "liveness",
            "valence",
        ]

    if year_col not in year_data.columns:
        raise ValueError(f"year_col '{year_col}' not found in DataFrame columns.")

    missing = [c for c in sound_features if c not in year_data.columns]
    if missing:
        raise ValueError(f"Missing feature columns in DataFrame: {missing}")

    # Ensure year column is numeric for correct ordering
    df = year_data.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col])
    df = df.sort_values(by=year_col)

    if title is None:
        title = "Trend of various sound features over time"

    fig = px.line(
        df,
        x=year_col,
        y=sound_features,
        title=title,
        markers=markers,
        template=template,
    )
    fig.update_layout(xaxis_title=year_col, yaxis_title="Feature value (mean)")

    if save_path:
        # save as standalone interactive HTML
        fig.write_html(save_path)

    if show:
        fig.show()

    return fig