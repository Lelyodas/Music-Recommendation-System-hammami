import matplotlib.pyplot as plt
from typing import Optional
from wordcloud import WordCloud

def plot_artist_wordcloud(
    wordcloud: WordCloud,
    figsize: tuple = (8, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Display an artist WordCloud using plt.imshow(wordcloud).
    - wordcloud: instance of wordcloud.WordCloud (e.g., from generate_artist_wordcloud).
    - figsize: figure size in inches.
    - title: optional title string.
    - save_path: if provided, the figure will be saved to this path.
    - show: whether to call plt.show(); if False the figure is closed and not shown.
    Returns the matplotlib Axes.
    """
    if not isinstance(wordcloud, WordCloud):
        raise ValueError("wordcloud must be an instance of wordcloud.WordCloud")

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax