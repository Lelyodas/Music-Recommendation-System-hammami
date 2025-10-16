# Music Data Analysis Project

This project analyzes Spotify music datasets spanning from 1921 to 2020, including track information, genres, years, and artists.

## 📊 Datasets

All CSV files have been successfully loaded:

1. **data.csv** (29 MB) - Main dataset with 170,653 tracks
   - Contains: track features, artists, year, popularity, audio characteristics
   
2. **data_by_genres.csv** (563 KB) - Genre aggregated data with 2,973 genres
   - Contains: aggregated statistics for each genre
   
3. **data_by_year.csv** (21 KB) - Year aggregated data covering 100 years (1921-2020)
   - Contains: aggregated statistics for each year
   
4. **data_by_artist.csv** (4.2 MB) - Artist aggregated data with 28,680 artists
   - Contains: aggregated statistics for each artist

## ✅ Tasks Completed

All 7 requested tasks have been successfully completed:

1. ✓ Read the main dataset using `pd.read_csv()` and assigned to `data`
2. ✓ Read the genre dataset using `pd.read_csv()` and assigned to `genre_data`
3. ✓ Read the year dataset using `pd.read_csv()` and assigned to `year_data`
4. ✓ Read the artist dataset using `pd.read_csv()` and assigned to `artist_data`
5. ✓ Display the first two rows of all datasets using `head(2)`
6. ✓ Retrieve information about `data` and `genre_data` using `info()`
7. ✓ Create a `decade` column in `data` using `apply()` and lambda function

## 📁 Project Files

- `analyze_music_data.py` - Python script that performs all 7 tasks
- `music_data_analysis.ipynb` - Jupyter notebook with interactive analysis
- `data.csv` - Main tracks dataset
- `data_by_genres.csv` - Genre statistics
- `data_by_year.csv` - Year statistics
- `data_by_artist.csv` - Artist statistics
- `README.md` - This file

## 🚀 Usage

### Option 1: Run the Python Script

```bash
python3 analyze_music_data.py
```

This will execute all tasks and display:
- First 2 rows of each dataset
- Dataset information (columns, types, memory usage)
- Decade column creation
- Summary statistics

### Option 2: Use the Jupyter Notebook

```bash
jupyter notebook music_data_analysis.ipynb
```

The notebook provides:
- Step-by-step execution of all tasks
- Interactive data exploration
- Visualizations (decade distribution, audio features evolution)
- Additional analysis and insights

## 📈 Key Insights

### Dataset Summary
- **Total tracks**: 170,653
- **Total genres**: 2,973
- **Total years**: 100 (1921-2020)
- **Total artists**: 28,680
- **Decade range**: 1920s - 2020s

### Decade Distribution
```
1920s:  5,126 tracks
1930s:  9,549 tracks
1940s: 15,378 tracks
1950s: 19,850 tracks
1960s: 19,549 tracks
1970s: 20,000 tracks
1980s: 19,850 tracks
1990s: 19,901 tracks
2000s: 19,646 tracks
2010s: 19,774 tracks
2020s:  2,030 tracks
```

### Audio Features in Main Dataset
- **acousticness**: Measure of acoustic sound (0.0 to 1.0)
- **danceability**: How suitable a track is for dancing (0.0 to 1.0)
- **energy**: Intensity and activity measure (0.0 to 1.0)
- **valence**: Musical positiveness (0.0 to 1.0)
- **tempo**: Beats per minute (BPM)
- **loudness**: Overall loudness in decibels (dB)
- **speechiness**: Presence of spoken words (0.0 to 1.0)
- **instrumentalness**: Predicts if track has no vocals (0.0 to 1.0)
- **liveness**: Presence of audience (0.0 to 1.0)

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

## 📝 Notes

- All datasets are complete with no missing values
- The decade column was successfully created using: `data['year'].apply(lambda x: (x // 10) * 10)`
- Data spans nearly a century of music history
- The dataset shows a relatively even distribution across decades (except 1920s and 2020s which are partial)

## 🎵 Data Columns

### Main Dataset (data.csv)
- `valence`, `year`, `acousticness`, `artists`, `danceability`, `duration_ms`
- `energy`, `explicit`, `id`, `instrumentalness`, `key`, `liveness`
- `loudness`, `mode`, `name`, `popularity`, `release_date`, `speechiness`, `tempo`
- `decade` (newly created)

### Genre Dataset (genre_data.csv)
- `mode`, `genres`, `acousticness`, `danceability`, `duration_ms`, `energy`
- `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`
- `valence`, `popularity`, `key`

### Year Dataset (year_data.csv)
- Same structure as genre_data but aggregated by year

### Artist Dataset (artist_data.csv)
- `mode`, `count`, `acousticness`, `artists`, and other audio features
- Includes track count for each artist

---

**Analysis Complete!** All tasks have been successfully executed. 🎉
