import re
import pandas as pd

def _to_artist_list(cell):
	"""Normalize an artist cell into a list of artist strings."""
	if pd.isna(cell):
		return []
	if isinstance(cell, (list, tuple, set)):
		return [str(x).strip() for x in cell if x is not None and str(x).strip()]
	if isinstance(cell, str):
		# split common separators and also split featured artists
		parts = re.split(r"[,\|;]+", cell)
		subparts = []
		for p in parts:
			subparts.extend(re.split(r"\s+feat\.?\s+|\s+ft\.?\s+", p, flags=re.IGNORECASE))
		return [p.strip() for p in subparts if p.strip()]
	return [str(cell).strip()]

def top_artists_by_song_count(df: pd.DataFrame, artist_col: str = "artists", top_n: int = 10) -> pd.DataFrame:
	"""
	Return a DataFrame of top_n artists with the most songs produced.
	Columns: 'count' (number of songs) and 'artists' (artist name).
	"""
	if artist_col not in df.columns:
		raise ValueError(f"DataFrame must contain a '{artist_col}' column.")

	tmp = df[[artist_col]].copy()
	tmp["_artist_list"] = tmp[artist_col].apply(_to_artist_list)
	tmp = tmp.explode("_artist_list").dropna(subset=["_artist_list"])
	tmp["artist"] = tmp["_artist_list"].astype(str).str.strip()
	if tmp.empty:
		return pd.DataFrame(columns=["count", "artists"])

	counts = tmp.groupby("artist").size().reset_index(name="count")
	counts = counts.sort_values("count", ascending=False).reset_index(drop=True)
	top = counts.head(top_n).copy()
	top = top.rename(columns={"artist": "artists"})[["count", "artists"]]
	return top

# Compute and assign the top-10 most-song-produced artists (variable expected by the prompt)
# Example usage assuming DataFrame is named `tracks`:
# top10_most_song_produced_artists = top_artists_by_song_count(tracks, artist_col='artists', top_n=10)