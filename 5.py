import re
from typing import Any, Iterable, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def _to_genre_list(cell: Any) -> List[str]:
	"""Normalize a genre cell into a list of genre strings."""
	if pd.isna(cell):
		return []
	if isinstance(cell, (list, tuple, set)):
		return [str(x).strip() for x in cell if x is not None and str(x).strip()]
	if isinstance(cell, str):
		# split common separators like commas, pipes, semicolons
		parts = re.split(r"[,\|;]+", cell)
		return [p.strip() for p in parts if p.strip()]
	return [str(cell).strip()]

def generate_genre_wordcloud(
	data: pd.DataFrame,
	genre_col: str = "genres",
	stopwords: Optional[Iterable[str]] = None,
	width: int = 800,
	height: int = 800,
	background_color: str = "white",
	max_words: int = 40,
	min_font_size: int = 10,
	save_path: Optional[str] = None,
	show: bool = True,
) -> WordCloud:
	"""
	Generate a WordCloud from the genres present in `data[genre_col]`.
	Returns the WordCloud object.
	"""
	if genre_col not in data.columns:
		raise ValueError(f"DataFrame must contain a '{genre_col}' column.")

	# Normalize and collect all genres
	series = data[genre_col].dropna().map(_to_genre_list)
	flat_genres = [g for sub in series for g in sub]
	if not flat_genres:
		raise ValueError("No genres found to build word cloud.")

	# Build a single text string for the WordCloud
	comment_words = " ".join(flat_genres)

	# Compose stopwords
	_stopwords = set(STOPWORDS)
	if stopwords:
		_stopwords.update(stopwords)

	# Generate word cloud
	wc = WordCloud(
		width=width,
		height=height,
		background_color=background_color,
		stopwords=_stopwords,
		max_words=max_words,
		min_font_size=min_font_size,
	).generate(comment_words)

	# Plot
	plt.figure(figsize=(width / 100, height / 100))
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")
	plt.tight_layout(pad=0)

	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches="tight")

	if show:
		plt.show()
	else:
		plt.close()

	return wc