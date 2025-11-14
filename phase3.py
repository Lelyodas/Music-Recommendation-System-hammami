# phase3/cluster_analysis.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ...existing code to load genre_df and song_df...

# 1. Fit K-means on genre data (12 clusters)
genre_features = genre_df.select_dtypes(include=['float64', 'int64'])  # select numeric columns
kmeans_genre = KMeans(n_clusters=12, random_state=42)
genre_df['cluster'] = kmeans_genre.fit_predict(genre_features)

# 2. Visualize genre clusters with t-SNE
tsne = TSNE(n_components=2, random_state=42)
genre_tsne = tsne.fit_transform(genre_features)
genre_df['tsne_1'] = genre_tsne[:, 0]
genre_df['tsne_2'] = genre_tsne[:, 1]

fig1 = px.scatter(
    genre_df, x='tsne_1', y='tsne_2', color='cluster',
    hover_data=['genre'] if 'genre' in genre_df.columns else None,
    title='t-SNE Visualization of Genre Clusters'
)
fig1.show()

# 3. Fit K-means on song data (25 clusters)
song_features = song_df.select_dtypes(include=['float64', 'int64'])
kmeans_song = KMeans(n_clusters=25, random_state=42)
song_df['cluster'] = kmeans_song.fit_predict(song_features)

# 4. Visualize song clusters with PCA
pca = PCA(n_components=2, random_state=42)
song_pca = pca.fit_transform(song_features)
song_df['pca_1'] = song_pca[:, 0]
song_df['pca_2'] = song_pca[:, 1]

fig2 = px.scatter(
    song_df, x='pca_1', y='pca_2', color='cluster',
    hover_data=['song_name', 'artist'] if 'song_name' in song_df.columns and 'artist' in song_df.columns else None,
    title='PCA Visualization of Song Clusters'
)
fig2.show()