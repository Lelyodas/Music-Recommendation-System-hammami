#!/usr/bin/env python3
"""
Music Data Analysis Script
This script loads and analyzes multiple music datasets
"""

import pandas as pd
import numpy as np

print("="*80)
print("MUSIC DATA ANALYSIS")
print("="*80)

# 1. Read the main dataset
print("\n1. Reading main dataset (data.csv)...")
data = pd.read_csv('data.csv')
print("   ✓ Loaded successfully!")

# 2. Read the genre dataset
print("\n2. Reading genre dataset (data_by_genres.csv)...")
genre_data = pd.read_csv('data_by_genres.csv')
print("   ✓ Loaded successfully!")

# 3. Read the year dataset
print("\n3. Reading year dataset (data_by_year.csv)...")
year_data = pd.read_csv('data_by_year.csv')
print("   ✓ Loaded successfully!")

# 4. Read the artist dataset
print("\n4. Reading artist dataset (data_by_artist.csv)...")
artist_data = pd.read_csv('data_by_artist.csv')
print("   ✓ Loaded successfully!")

# 5. Display the first two rows of each dataset
print("\n" + "="*80)
print("5. DISPLAYING FIRST TWO ROWS OF EACH DATASET")
print("="*80)

print("\n--- DATA (Main Dataset) ---")
print(data.head(2))

print("\n--- GENRE_DATA ---")
print(genre_data.head(2))

print("\n--- YEAR_DATA ---")
print(year_data.head(2))

print("\n--- ARTIST_DATA ---")
print(artist_data.head(2))

# 6. Retrieve information about data and genre_data
print("\n" + "="*80)
print("6. DATASET INFORMATION")
print("="*80)

print("\n--- DATA (Main Dataset) INFO ---")
print(data.info())

print("\n--- GENRE_DATA INFO ---")
print(genre_data.info())

# 7. Create a decade column in data
print("\n" + "="*80)
print("7. CREATING DECADE COLUMN")
print("="*80)

print("\nAdding 'decade' column to main dataset...")
data['decade'] = data['year'].apply(lambda x: (x // 10) * 10)
print("   ✓ Decade column created successfully!")

print("\nFirst 10 rows showing year and decade:")
print(data[['name', 'year', 'decade']].head(10))

print("\nDecade distribution:")
decade_counts = data['decade'].value_counts().sort_index()
print(decade_counts)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

# Summary statistics
print("\n--- DATASET SUMMARY ---")
print(f"Total tracks in main dataset: {len(data):,}")
print(f"Total genres: {len(genre_data):,}")
print(f"Total years covered: {len(year_data):,}")
print(f"Total artists: {len(artist_data):,}")
print(f"Year range: {data['year'].min()} - {data['year'].max()}")
print(f"Decade range: {data['decade'].min()} - {data['decade'].max()}")
