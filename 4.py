import re
from typing import List, Optional, Iterable, Any
import pandas as pd
import plotly.express as px

def _to_genre_list(cell: Any) -> List[str]:
	"""Normalize a genre cell into a list of genre strings."""
	if pd.isna(cell):
		return []
	if isinstance(cell, (list, tuple, set)):
		return [str(x).strip() for x in cell if x is not None and str(x).strip()]
	if isinstance(cell, str):
		# split common separators
		parts = re.split(r"[,\|;]+", cell)
		return [p.strip() for p in parts if p.strip()]
	# fallback
	return [str(cell).strip()]

def plot_top_genre_features(
	data: pd.DataFrame,
	genre_col: str = "genres",
	popularity_col: str = "popularity",
	features: Optional[Iterable[str]] = None,
	top_n: int = 10,
	title: Optional[str] = None,
	template: str = "plotly_white",
	save_path: Optional[str] = None,
	show: bool = True,
) -> "plotly.graph_objs._figure.Figure":
	"""
	Identify top N genres by mean popularity and plot grouped bar chart of feature means.
	- data: DataFrame containing genre, popularity and feature columns.
	- genre_col: column with genre info (list or string).
	- popularity_col: numeric popularity column used to rank genres.
	- features: iterable of sound feature column names; defaults to common ones.
	- Returns the plotly Figure.
	"""
	if features is None:
		features = ["valence", "energy", "danceability", "acousticness"]

	cols_required = {genre_col, popularity_col} | set(features)
	missing = [c for c in cols_required if c not in data.columns]
	if missing:
		raise ValueError(f"Missing required columns in DataFrame: {missing}")

	# Prepare exploded genre DataFrame
	df = data[[genre_col, popularity_col] + list(features)].copy()
	df["_genre_list"] = df[genre_col].apply(_to_genre_list)
	df = df.explode("_genre_list").rename(columns={"_genre_list": "genre"})
	df = df.dropna(subset=["genre"])
	df["genre"] = df["genre"].astype(str).str.strip()

	if df.empty:
		raise ValueError("No genres found after normalization/explosion.")

	# Determine top genres by mean popularity
	genre_pop = df.groupby("genre")[popularity_col].mean()
	top_genres = genre_pop.nlargest(top_n).index.tolist()
	if not top_genres:
		raise ValueError("Unable to determine top genres.")

	# Aggregate mean feature values for top genres and preserve ordering
	agg = (
		df[df["genre"].isin(top_genres)]
		.groupby("genre")[list(features)]
		.mean()
		.reset_index()
	)
	# ensure consistent ordering
	agg["genre"] = pd.Categorical(agg["genre"], categories=top_genres, ordered=True)
	agg = agg.sort_values("genre")

	if title is None:
		title = f"Trend of various sound features over top {len(top_genres)} genres"

	fig = px.bar(
		agg,
		x="genre",
		y=list(features),
		barmode="group",
		title=title,
		template=template,
	)
	fig.update_layout(xaxis_title="Genre", yaxis_title="Mean feature value")

	if save_path:
		fig.write_html(save_path)

	if show:
		fig.show()

	return fig