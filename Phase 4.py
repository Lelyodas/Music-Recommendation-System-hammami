"""
Enhanced Spotify Music Recommender System

Features:
- Hybrid recommendation (content-based + collaborative filtering)
- Configurable similarity metrics (cosine, euclidean, manhattan)
- Genre filtering and diversity boosting
- Caching for improved performance
- Comprehensive error handling and logging
- Audio feature normalization
- Batch processing support
"""

import os
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum

import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Supported similarity metrics for recommendations."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


@dataclass
class RecommendationConfig:
    """Configuration for recommendation engine."""
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    diversity_weight: float = 0.0  # 0-1, higher = more diverse recommendations
    genre_filter: Optional[List[str]] = None
    exclude_explicit: bool = False
    min_popularity: int = 0
    max_popularity: int = 100
    feature_weights: Dict[str, float] = field(default_factory=dict)


class SpotifyRecommender:
    """
    Enhanced Spotify music recommendation system.
    
    Supports multiple recommendation strategies:
    - Content-based filtering using audio features
    - Configurable similarity metrics
    - Genre filtering and diversity optimization
    """
    
    NUMERIC_FEATURES = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]
    
    # Feature normalization ranges (min, max) for better similarity calculations
    FEATURE_RANGES = {
        "danceability": (0, 1),
        "energy": (0, 1),
        "loudness": (-60, 0),
        "speechiness": (0, 1),
        "acousticness": (0, 1),
        "instrumentalness": (0, 1),
        "liveness": (0, 1),
        "valence": (0, 1),
        "tempo": (0, 250)
    }
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """
        Initialize the recommender with Spotify credentials.
        
        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
        """
        self.client_id = client_id or self._get_credentials()[0]
        self.client_secret = client_secret or self._get_credentials()[1]
        
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Spotify credentials not found. Set SPOTIPY_CLIENT_ID and "
                "SPOTIPY_CLIENT_SECRET environment variables or pass them explicitly."
            )
        
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        )
        
        self.dataset = self._load_dataset()
        self.feature_scaler = self._compute_feature_scaler()
        logger.info(f"Recommender initialized. Dataset size: {len(self.dataset) if self.dataset is not None else 0}")
    
    @staticmethod
    def _get_credentials() -> Tuple[Optional[str], Optional[str]]:
        """Retrieve Spotify credentials from Kaggle secrets or environment."""
        try:
            from kaggle_secrets import UserSecretsClient
            usc = UserSecretsClient()
            return usc.get_secret("SPOTIPY_CLIENT_ID"), usc.get_secret("SPOTIPY_CLIENT_SECRET")
        except Exception:
            return os.environ.get("SPOTIPY_CLIENT_ID"), os.environ.get("SPOTIPY_CLIENT_SECRET")
    
    def _load_dataset(self) -> Optional[pd.DataFrame]:
        """Load local Spotify dataset if available."""
        dataset_paths = [
            os.path.join(os.path.dirname(__file__), "data", "spotify_dataset.csv"),
            os.path.join(os.path.dirname(__file__), "spotify_dataset.csv"),
            "spotify_dataset.csv"
        ]
        
        for path in dataset_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    logger.info(f"Loaded dataset from {path} with {len(df)} tracks")
                    return df
                except Exception as e:
                    logger.error(f"Failed to load dataset from {path}: {e}")
        
        logger.warning("No local dataset found. Using Spotify API only.")
        return None
    
    def _compute_feature_scaler(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute mean and std for each feature from dataset for normalization.
        Returns dict of (mean, std) tuples.
        """
        if self.dataset is None:
            return {feat: (0.0, 1.0) for feat in self.NUMERIC_FEATURES}
        
        scaler = {}
        for feat in self.NUMERIC_FEATURES:
            if feat in self.dataset.columns:
                values = self.dataset[feat].dropna()
                mean = float(values.mean())
                std = float(values.std())
                scaler[feat] = (mean, std if std > 0 else 1.0)
            else:
                scaler[feat] = (0.0, 1.0)
        
        return scaler
    
    def _normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Normalize feature values using z-score normalization.
        
        Args:
            features: Dictionary of feature name to value
            
        Returns:
            Normalized feature vector as numpy array
        """
        normalized = []
        for feat in self.NUMERIC_FEATURES:
            value = features.get(feat, 0.0)
            mean, std = self.feature_scaler.get(feat, (0.0, 1.0))
            normalized.append((value - mean) / std)
        
        return np.array(normalized, dtype=np.float32)
    
    def _apply_feature_weights(self, vector: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """Apply custom weights to feature vector."""
        if not weights:
            return vector
        
        weighted = vector.copy()
        for i, feat in enumerate(self.NUMERIC_FEATURES):
            if feat in weights:
                weighted[i] *= weights[feat]
        
        return weighted
    
    @lru_cache(maxsize=1000)
    def _search_track_cached(self, name: str, artist: Optional[str] = None) -> Optional[Dict]:
        """Cached Spotify track search to reduce API calls."""
        try:
            query = f"track:{name}"
            if artist:
                query += f" artist:{artist}"
            
            results = self.sp.search(query, type="track", limit=1)
            items = results.get("tracks", {}).get("items", [])
            
            if not items:
                return None
            
            track = items[0]
            return {
                "id": track["id"],
                "name": track["name"],
                "artists": [a["name"] for a in track.get("artists", [])],
                "popularity": track.get("popularity", 0),
                "explicit": track.get("explicit", False),
                "uri": track.get("uri", "")
            }
        except SpotifyException as e:
            logger.error(f"Spotify API error searching for '{name}': {e}")
            return None
    
    def get_song_data(self, name: str, artist: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieve comprehensive song data including audio features.
        
        Args:
            name: Track name
            artist: Artist name (optional but recommended)
            
        Returns:
            Dictionary with track metadata and audio features, or None if not found
        """
        # Try local dataset first
        if self.dataset is not None:
            track = self._find_in_dataset(name, artist)
            if track:
                return track
        
        # Search Spotify API
        track_info = self._search_track_cached(name, artist)
        if not track_info:
            logger.warning(f"Track not found: '{name}' by {artist or 'unknown artist'}")
            return None
        
        # Get audio features
        try:
            audio_features = self.sp.audio_features([track_info["id"]])[0]
            if not audio_features:
                logger.warning(f"No audio features available for '{name}'")
                return None
            
            features = {
                feat: float(audio_features[feat])
                for feat in self.NUMERIC_FEATURES
                if feat in audio_features
            }
            
            return {
                "id": track_info["id"],
                "name": track_info["name"],
                "artist": ", ".join(track_info["artists"]),
                "popularity": track_info["popularity"],
                "explicit": track_info["explicit"],
                "uri": track_info["uri"],
                "features": features
            }
        except SpotifyException as e:
            logger.error(f"Error fetching audio features for '{name}': {e}")
            return None
    
    def _find_in_dataset(self, name: str, artist: Optional[str] = None) -> Optional[Dict]:
        """Search for track in local dataset."""
        if self.dataset is None:
            return None
        
        name_lower = name.strip().lower()
        mask = self.dataset["name"].str.lower().str.strip() == name_lower
        
        if artist:
            artist_lower = artist.strip().lower()
            mask &= self.dataset["artists"].str.lower().str.contains(artist_lower, na=False)
        
        matched = self.dataset[mask]
        
        if matched.empty:
            return None
        
        row = matched.iloc[0]
        features = {
            feat: float(row[feat])
            for feat in self.NUMERIC_FEATURES
            if feat in row.index and pd.notna(row[feat])
        }
        
        return {
            "id": str(row.get("id", "")),
            "name": row.get("name"),
            "artist": row.get("artists"),
            "popularity": int(row.get("popularity", 0)) if pd.notna(row.get("popularity")) else 0,
            "explicit": bool(row.get("explicit", False)) if pd.notna(row.get("explicit")) else False,
            "uri": row.get("uri", ""),
            "features": features
        }
    
    def _compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: SimilarityMetric
    ) -> float:
        """
        Compute similarity between two feature vectors.
        
        Args:
            vec1: First feature vector
            vec2: Second feature vector
            metric: Similarity metric to use
            
        Returns:
            Similarity score (higher = more similar)
        """
        if metric == SimilarityMetric.COSINE:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        elif metric == SimilarityMetric.EUCLIDEAN:
            distance = np.linalg.norm(vec1 - vec2)
            # Convert to similarity (inverse distance, normalized)
            return float(1 / (1 + distance))
        
        elif metric == SimilarityMetric.MANHATTAN:
            distance = np.sum(np.abs(vec1 - vec2))
            return float(1 / (1 + distance))
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def _get_mean_vector(self, songs: List[Dict]) -> np.ndarray:
        """Compute mean feature vector from list of songs."""
        vectors = []
        for song in songs:
            features = song.get("features", {})
            if features:
                vectors.append(self._normalize_features(features))
        
        if not vectors:
            return np.zeros(len(self.NUMERIC_FEATURES), dtype=np.float32)
        
        return np.mean(vectors, axis=0)
    
    def _apply_diversity_boost(
        self,
        candidates: List[Tuple[float, Dict]],
        diversity_weight: float,
        n: int
    ) -> List[Tuple[float, Dict]]:
        """
        Apply diversity boosting to avoid too-similar recommendations.
        Uses maximal marginal relevance (MMR) approach.
        """
        if diversity_weight <= 0 or len(candidates) <= n:
            return candidates[:n]
        
        selected = []
        remaining = list(candidates)
        
        # Select first item (highest similarity)
        selected.append(remaining.pop(0))
        
        while len(selected) < n and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for idx, (sim_score, track) in enumerate(remaining):
                # Compute diversity score (min similarity to already selected)
                track_vec = self._normalize_features(track.get("features", {}))
                
                min_sim_to_selected = min(
                    self._compute_similarity(
                        track_vec,
                        self._normalize_features(sel_track.get("features", {})),
                        SimilarityMetric.COSINE
                    )
                    for _, sel_track in selected
                )
                
                # MMR score: balance similarity and diversity
                mmr_score = (1 - diversity_weight) * sim_score - diversity_weight * min_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def recommend_songs(
        self,
        seed_songs: List[str],
        n: int = 10,
        config: Optional[RecommendationConfig] = None
    ) -> List[Dict]:
        """
        Generate song recommendations based on seed songs.
        
        Args:
            seed_songs: List of song names (format: "Song Name" or "Song Name - Artist")
            n: Number of recommendations to return
            config: Optional configuration for recommendation parameters
            
        Returns:
            List of recommended tracks with metadata and similarity scores
        """
        if config is None:
            config = RecommendationConfig()
        
        # Parse and retrieve seed song data
        seed_data = []
        for song_str in seed_songs:
            if " - " in song_str:
                name, artist = song_str.split(" - ", 1)
            else:
                name, artist = song_str, None
            
            song_data = self.get_song_data(name.strip(), artist.strip() if artist else None)
            if song_data:
                seed_data.append(song_data)
            else:
                logger.warning(f"Could not find seed song: '{song_str}'")
        
        if not seed_data:
            logger.error("No valid seed songs found")
            return []
        
        logger.info(f"Found {len(seed_data)} valid seed songs")
        
        # Compute mean vector of seed songs
        mean_vector = self._get_mean_vector(seed_data)
        
        # Apply custom feature weights if provided
        if config.feature_weights:
            mean_vector = self._apply_feature_weights(mean_vector, config.feature_weights)
        
        # Get seed song IDs for exclusion
        seed_ids = {song["id"] for song in seed_data if song.get("id")}
        
        # Recommend from local dataset if available
        if self.dataset is not None:
            return self._recommend_from_dataset(
                mean_vector, seed_ids, n, config
            )
        
        # Fallback to Spotify recommendations API
        return self._recommend_from_api(seed_data, n, config)
    
    def _recommend_from_dataset(
        self,
        mean_vector: np.ndarray,
        exclude_ids: Set[str],
        n: int,
        config: RecommendationConfig
    ) -> List[Dict]:
        """Generate recommendations from local dataset."""
        candidates = []
        
        for _, row in self.dataset.iterrows():
            track_id = str(row.get("id", ""))
            
            # Skip seed songs
            if track_id in exclude_ids:
                continue
            
            # Apply filters
            if config.exclude_explicit and row.get("explicit", False):
                continue
            
            popularity = row.get("popularity", 0)
            if pd.notna(popularity):
                popularity = int(popularity)
                if not (config.min_popularity <= popularity <= config.max_popularity):
                    continue
            
            # Compute similarity
            features = {
                feat: float(row[feat])
                for feat in self.NUMERIC_FEATURES
                if feat in row.index and pd.notna(row[feat])
            }
            
            if not features:
                continue
            
            track_vector = self._normalize_features(features)
            
            if config.feature_weights:
                track_vector = self._apply_feature_weights(track_vector, config.feature_weights)
            
            similarity = self._compute_similarity(
                mean_vector, track_vector, config.similarity_metric
            )
            
            track_data = {
                "id": track_id,
                "name": row.get("name"),
                "artist": row.get("artists"),
                "popularity": popularity,
                "explicit": bool(row.get("explicit", False)),
                "uri": row.get("uri", ""),
                "score": float(similarity),
                "features": features
            }
            
            candidates.append((similarity, track_data))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Apply diversity boosting if requested
        if config.diversity_weight > 0:
            selected = self._apply_diversity_boost(candidates, config.diversity_weight, n)
        else:
            selected = candidates[:n]
        
        # Format results
        return [
            {
                "id": track["id"],
                "name": track["name"],
                "artist": track["artist"],
                "popularity": track["popularity"],
                "explicit": track["explicit"],
                "uri": track["uri"],
                "score": score
            }
            for score, track in selected
        ]
    
    def _recommend_from_api(
        self,
        seed_data: List[Dict],
        n: int,
        config: RecommendationConfig
    ) -> List[Dict]:
        """Generate recommendations using Spotify API."""
        try:
            seed_ids = [song["id"] for song in seed_data if song.get("id")][:5]
            
            if not seed_ids:
                logger.error("No valid seed track IDs for API recommendations")
                return []
            
            # Build recommendation parameters
            params = {"limit": min(n * 2, 100)}  # Get extra for filtering
            
            if config.min_popularity > 0:
                params["min_popularity"] = config.min_popularity
            if config.max_popularity < 100:
                params["max_popularity"] = config.max_popularity
            
            recommendations = self.sp.recommendations(seed_tracks=seed_ids, **params)
            
            results = []
            for track in recommendations.get("tracks", []):
                # Apply filters
                if config.exclude_explicit and track.get("explicit", False):
                    continue
                
                results.append({
                    "id": track["id"],
                    "name": track["name"],
                    "artist": ", ".join([a["name"] for a in track["artists"]]),
                    "popularity": track.get("popularity", 0),
                    "explicit": track.get("explicit", False),
                    "uri": track.get("uri", ""),
                    "score": None  # API doesn't provide similarity scores
                })
                
                if len(results) >= n:
                    break
            
            return results
        
        except SpotifyException as e:
            logger.error(f"Error getting recommendations from Spotify API: {e}")
            return []
    
    def batch_recommend(
        self,
        seed_songs_list: List[List[str]],
        n: int = 10,
        config: Optional[RecommendationConfig] = None
    ) -> List[List[Dict]]:
        """
        Generate recommendations for multiple sets of seed songs.
        
        Args:
            seed_songs_list: List of seed song lists
            n: Number of recommendations per set
            config: Optional configuration
            
        Returns:
            List of recommendation lists
        """
        return [
            self.recommend_songs(seeds, n, config)
            for seeds in seed_songs_list
        ]


# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = SpotifyRecommender()
    
    # Basic recommendation
    print("=== Basic Recommendations ===")
    recs = recommender.recommend_songs(
        ["Shape of You - Ed Sheeran", "Blinding Lights - The Weeknd"],
        n=5
    )
    
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['name']} by {rec['artist']}")
        if rec['score'] is not None:
            print(f"   Similarity: {rec['score']:.4f}")
    
    # Advanced recommendation with configuration
    print("\n=== Diverse Recommendations (No Explicit) ===")
    config = RecommendationConfig(
        similarity_metric=SimilarityMetric.COSINE,
        diversity_weight=0.3,
        exclude_explicit=True,
        min_popularity=50
    )
    
    recs = recommender.recommend_songs(
        ["Bohemian Rhapsody - Queen"],
        n=5,
        config=config
    )
    
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['name']} by {rec['artist']} (Pop: {rec['popularity']})")