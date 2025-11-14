import re
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _decade_sort_key(val: str) -> int:
	"""Extract first integer from a decade string for sorting (e.g. '1980s' -> 1980)."""
	m = re.search(r"\d+", str(val))
	return int(m.group()) if m else float("inf")

def plot_decade_count(
	data: pd.DataFrame,
	figsize: tuple = (10, 6),
	order: Optional[Sequence] = None,
	palette: Optional[Iterable] = None,
	title: str = "Distribution of tracks by decade",
	xlabel: str = "Decade",
	ylabel: str = "Number of tracks",
	save_path: Optional[str] = None,
	show: bool = True,
) -> plt.Axes:
	"""
	Draw a count plot of data['decade'] using seaborn.countplot.
	- data: DataFrame containing a 'decade' column (e.g. '1970s', '1980s').
	- order: optional list specifying the order of decades; if None it is inferred and sorted numerically.
	- palette: seaborn/matplotlib palette spec.
	- save_path: if provided, the figure will be saved to this path.
	- returns the matplotlib Axes.
	"""
	if "decade" not in data.columns:
		raise ValueError("DataFrame must contain a 'decade' column.")

	# Infer order if not provided: sort decades numerically (handles '1970s', '1980s', '1990', etc.)
	if order is None:
		unique_vals = data["decade"].dropna().unique()
		order = sorted(unique_vals, key=_decade_sort_key)

	plt.figure(figsize=figsize)
	sns.set_style("whitegrid")
	ax = sns.countplot(x="decade", data=data, order=order, palette=palette)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches="tight")

	if show:
		plt.show()

	return ax