"""
Module: network_preprocessor.py
===============================

Preprocess network event data for K-means clustering and LSA analysis.

The preprocessing pipeline normalizes numerical features, vectorizes log
messages with TF-IDF, reduces text dimensionality with LSA, and concatenates
all features for clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import re

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


@dataclass
class NetworkDataConfig:
    """Configuration for network event preprocessing."""
    # TF-IDF settings
    tfidf_max_df: float = 0.95
    tfidf_min_df: int = 2
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_max_features: int = 1000

    # LSA settings
    lsa_n_components: int = 20

    # Numerical features
    log_transform_features: List[str] = None

    def __post_init__(self):
        if self.log_transform_features is None:
            self.log_transform_features = ["duration_ms", "bytes_transferred"]


class NetworkPreprocessor:
    """
    Preprocessor for network event data.

    Handles the complete preprocessing pipeline for unsupervised network event
    classification, combining numerical feature engineering, TF-IDF text
    extraction, and LSA dimensionality reduction.

    Attributes:
        config: Preprocessing configuration
        scaler: StandardScaler for numerical features
        tfidf: TfidfVectorizer for log messages
        svd: TruncatedSVD for LSA
        fitted: Whether the preprocessor has been fitted
    """

    def __init__(self, config: Optional[NetworkDataConfig] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or NetworkDataConfig()
        self.scaler: Optional[StandardScaler] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.fitted = False

        self.numerical_features = [
            "duration_ms",
            "bytes_transferred",
            "port",
            "hour_of_day",
            "day_of_week",
        ]
        self.text_feature = "log_message"
        self.feature_names: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fit the preprocessor and transform data.

        Args:
            df: Raw network events DataFrame

        Returns:
            Tuple of (feature_matrix, processed_df)
        """
        numerical_df = self._engineer_numerical_features(df)

        self.scaler = StandardScaler()
        numerical_scaled = self.scaler.fit_transform(
            numerical_df[self.numerical_features].values
        )

        text_features = self._fit_transform_text(df[self.text_feature].values)

        combined_features = np.hstack([numerical_scaled, text_features])

        self.feature_names = (
            self.numerical_features +
            [f"lsa_component_{i}" for i in range(self.config.lsa_n_components)]
        )

        self.fitted = True

        processed_df = df.copy()
        for i, name in enumerate(self.feature_names):
            processed_df[name] = combined_features[:, i]

        return combined_features, processed_df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted transformers.

        Args:
            df: DataFrame to transform

        Returns:
            Combined feature matrix

        Raises:
            ValueError: If preprocessor hasn't been fitted
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        numerical_df = self._engineer_numerical_features(df)
        numerical_scaled = self.scaler.transform(
            numerical_df[self.numerical_features].values
        )

        tfidf_features = self.tfidf.transform(df[self.text_feature].values)
        text_features = self.svd.transform(tfidf_features)

        return np.hstack([numerical_scaled, text_features])

    def _engineer_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer numerical features.

        Applies log transform to heavy-tailed distributions (duration, bytes)
        to make them more suitable for K-means clustering.

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with engineered features
        """
        result = df.copy()

        for col in self.config.log_transform_features:
            if col in result.columns:
                result[col] = np.log1p(result[col])

        return result

    def _fit_transform_text(self, texts: np.ndarray) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and LSA, then transform text.

        Args:
            texts: Array of log message strings

        Returns:
            LSA feature matrix (n_samples, n_components)
        """
        cleaned_texts = [self._clean_text(text) for text in texts]

        self.tfidf = TfidfVectorizer(
            max_df=self.config.tfidf_max_df,
            min_df=self.config.tfidf_min_df,
            ngram_range=self.config.tfidf_ngram_range,
            max_features=self.config.tfidf_max_features,
            stop_words="english",
        )
        tfidf_matrix = self.tfidf.fit_transform(cleaned_texts)

        self.svd = TruncatedSVD(
            n_components=self.config.lsa_n_components,
            random_state=42,
        )
        lsa_features = self.svd.fit_transform(tfidf_matrix)

        explained_var = self.svd.explained_variance_ratio_.sum()
        print(f"LSA explained variance: {explained_var:.2%}")

        return lsa_features

    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess a log message.

        Args:
            text: Raw log message

        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r'\d+\.\d+\.\d+\.\d+', 'IP_ADDR', text)
        text = re.sub(r'\b\d+\b', 'NUM', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def get_feature_names(self) -> List[str]:
        """
        Get names of all features in the combined feature matrix.

        Returns:
            List of feature names
        """
        return self.feature_names

    def get_tfidf_vocab(self) -> Dict[str, int]:
        """
        Get TF-IDF vocabulary (term -> index mapping).

        Returns:
            Dictionary of terms to indices
        """
        if self.tfidf is None:
            return {}
        return self.tfidf.vocabulary_

    def get_top_terms_per_component(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Get top terms for each LSA component.

        Args:
            n_terms: Number of top terms to return per component

        Returns:
            Dictionary mapping component index to list of top terms
        """
        if self.tfidf is None or self.svd is None:
            return {}

        terms = self.tfidf.get_feature_names_out()
        components = self.svd.components_

        top_terms = {}
        for i, component in enumerate(components):
            top_indices = np.argsort(np.abs(component))[-n_terms:][::-1]
            top_terms[i] = [terms[idx] for idx in top_indices]

        return top_terms

    def get_state_dict(self) -> Dict:
        """
        Get state dictionary for saving preprocessor.

        Returns:
            Dictionary containing all preprocessor state
        """
        return {
            "config": {
                "tfidf_max_df": self.config.tfidf_max_df,
                "tfidf_min_df": self.config.tfidf_min_df,
                "tfidf_ngram_range": self.config.tfidf_ngram_range,
                "tfidf_max_features": self.config.tfidf_max_features,
                "lsa_n_components": self.config.lsa_n_components,
            },
            "scaler_mean": self.scaler.mean_.tolist() if self.scaler else None,
            "scaler_std": self.scaler.scale_.tolist() if self.scaler else None,
            "tfidf_vocab": self.get_tfidf_vocab(),
            "svd_components": self.svd.components_.tolist() if self.svd else None,
            "numerical_features": self.numerical_features,
            "feature_names": self.feature_names,
        }


def analyze_cluster_text_patterns(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    preprocessor: NetworkPreprocessor,
    n_terms: int = 10
) -> Dict[int, Dict[str, List[str]]]:
    """
    Analyze text patterns within each cluster.

    Finds the most distinctive terms in each cluster's log messages,
    useful for labeling clusters semantically.

    Args:
        df: DataFrame with log_message column
        cluster_labels: Cluster assignments from K-means
        preprocessor: Fitted NetworkPreprocessor
        n_terms: Number of top terms per cluster

    Returns:
        Dictionary mapping cluster IDs to their top terms
    """
    if preprocessor.tfidf is None:
        return {}

    cluster_patterns = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_texts = df.loc[mask, "log_message"].values

        cleaned = [preprocessor._clean_text(t) for t in cluster_texts]
        tfidf_matrix = preprocessor.tfidf.transform(cleaned)

        avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        terms = preprocessor.tfidf.get_feature_names_out()
        top_indices = np.argsort(avg_tfidf)[-n_terms:][::-1]

        cluster_patterns[int(cluster_id)] = {
            "top_terms": [terms[idx] for idx in top_indices],
            "term_scores": [float(avg_tfidf[idx]) for idx in top_indices],
            "sample_count": int(mask.sum()),
        }

    return cluster_patterns
