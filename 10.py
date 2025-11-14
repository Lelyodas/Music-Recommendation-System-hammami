import re
import pandas as pd

def _to_artist_list(cell):
	"""Normalize an artist cell into a list of artist strings."""
	if pd.isna(cell):
		return []
	if isinstance(cell, (list, tuple, set)):
		return [str(x).strip() for x in cell if x is not None and str(x).strip()]
	if isinstance(cell, str):
		parts = re.split(r"[,\|;]+", cell)
		subparts = []
		for p in parts:
			subparts.extend(re.split(r"\s+feat\.?\s+|\s+ft\.?\s+", p, flags=re.IGNORECASE))
		return [p.strip() for p in subparts if p.strip()]
	return [str(cell).strip()]

def top_artists_by_popularity(df: pd.DataFrame, artist_col: str = "artists", popularity_col: str = "popularity", top_n: int = 10) -> pd.DataFrame:
	"""
	Return top_n artists by mean popularity.
	Result columns: ['popularity', 'artists'] sorted by popularity descending.
	"""
	if artist_col not in df.columns or popularity_col not in df.columns:
		raise ValueError(f"DataFrame must contain '{artist_col}' and '{popularity_col}' columns.")

	tmp = df[[artist_col, popularity_col]].copy()
	tmp["_artist_list"] = tmp[artist_col].apply(_to_artist_list)
	tmp = tmp.explode("_artist_list").dropna(subset=["_artist_list"])
	tmp["artist"] = tmp["_artist_list"].astype(str).str.strip()
	if tmp.empty:
		return pd.DataFrame(columns=["popularity", "artists"])

	# Aggregate mean popularity per artist
	pop_by_artist = tmp.groupby("artist")[popularity_col].mean().reset_index()
	pop_by_artist = pop_by_artist.rename(columns={popularity_col: "popularity", "artist": "artists"})
	pop_by_artist = pop_by_artist.sort_values("popularity", ascending=False).reset_index(drop=True)
	return pop_by_artist.head(top_n)

# Example usage (merge into existing flow; `tracks` should be your DataFrame variable):
# top10_popular_artists = top_artists_by_popularity(tracks, artist_col='artists', popularity_col='popularity', top_n=10)
# print(top10_popular_artists[['popularity','artists']].sort_values('popularity', ascending=False))