import re
from typing import Any, Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def _to_artist_list(cell: Any) -> List[str]:
	"""Normalize an artist cell into a list of artist strings."""
	if pd.isna(cell):
		return []
	if isinstance(cell, (list, tuple, set)):
		return [str(x).strip() for x in cell if x is not None and str(x).strip()]
	if isinstance(cell, str):
		# split common separators like commas, pipes, semicolons and "feat." patterns
		parts = re.split(r"[,\|;]+", cell)
		# also split "feat." / "ft." occurrences (keep main artist names as separate tokens)
		subparts = []
		for p in parts:
			subparts.extend(re.split(r"\s+feat\.?\s+|\s+ft\.?\s+", p, flags=re.IGNORECASE))
		return [p.strip() for p in subparts if p.strip()]
	return [str(cell).strip()]

def generate_artist_wordcloud(
	data: pd.DataFrame,
	artist_col: str = "artists",
	stopwords: Optional[Iterable[str]] = None,
	width: int = 800,
	height: int = 800,
	background_color: str = "white",
	max_words: int = 40,
	min_font_size: int = 10,
	min_word_length: int = 3,
	save_path: Optional[str] = None,
	show: bool = True,
) -> WordCloud:
	"""
	Generate a WordCloud from the artists present in `data[artist_col]`.
	Returns the WordCloud object.
	"""
	if artist_col not in data.columns:
		raise ValueError(f"DataFrame must contain a '{artist_col}' column.")

	series = data[artist_col].dropna().map(_to_artist_list)
	flat_artists = [a for sub in series for a in sub]
	# filter by minimum name length
	if min_word_length:
		flat_artists = [a for a in flat_artists if len(a) >= int(min_word_length)]

	if not flat_artists:
		raise ValueError("No artist names found to build word cloud after filtering.")

	comment_words = " ".join(flat_artists)

	_stopwords = set(STOPWORDS)
	if stopwords:
		_stopwords.update(stopwords)

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