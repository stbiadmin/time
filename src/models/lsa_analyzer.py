"""
Module: lsa_analyzer.py
=======================

Latent Semantic Analysis for network log text feature extraction.

LSA uses TF-IDF vectorization followed by Truncated SVD to extract
latent topics from text documents.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


@dataclass
class LSAConfig:
    """Configuration for LSA analysis."""
    n_components: int = 20
    tfidf_max_df: float = 0.95
    tfidf_min_df: int = 2
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_max_features: int = 1000
    random_state: int = 42


class LSAAnalyzer:
    """
    Latent Semantic Analysis for text feature extraction.

    Provides a complete LSA pipeline: text preprocessing, TF-IDF vectorization,
    Truncated SVD dimensionality reduction, and topic interpretation.

    Attributes:
        config: LSA configuration
        tfidf: Fitted TfidfVectorizer
        svd: Fitted TruncatedSVD
        fitted: Whether the analyzer has been fitted
    """

    def __init__(self, config: Optional[LSAConfig] = None):
        """
        Initialize the LSA analyzer.

        Args:
            config: LSA configuration (uses defaults if None)
        """
        self.config = config or LSAConfig()
        self.tfidf: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the LSA model and transform texts to feature vectors.

        Args:
            texts: List of text documents (log messages)

        Returns:
            LSA feature matrix (n_samples, n_components)
        """
        cleaned_texts = [self._preprocess_text(text) for text in texts]

        self.tfidf = TfidfVectorizer(
            max_df=self.config.tfidf_max_df,
            min_df=self.config.tfidf_min_df,
            ngram_range=self.config.tfidf_ngram_range,
            max_features=self.config.tfidf_max_features,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )

        tfidf_matrix = self.tfidf.fit_transform(cleaned_texts)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.tfidf.vocabulary_)}")

        self.svd = TruncatedSVD(
            n_components=self.config.n_components,
            random_state=self.config.random_state,
        )

        lsa_features = self.svd.fit_transform(tfidf_matrix)

        explained_var = self.svd.explained_variance_ratio_.sum()
        print(f"LSA explained variance: {explained_var:.2%}")
        print(f"LSA features shape: {lsa_features.shape}")

        self.fitted = True
        return lsa_features

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using fitted LSA model.

        Args:
            texts: List of text documents

        Returns:
            LSA feature matrix

        Raises:
            ValueError: If analyzer hasn't been fitted
        """
        if not self.fitted:
            raise ValueError("LSA analyzer must be fitted before transform")

        cleaned_texts = [self._preprocess_text(text) for text in texts]
        tfidf_matrix = self.tfidf.transform(cleaned_texts)
        return self.svd.transform(tfidf_matrix)

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess a text document.

        Args:
            text: Raw text document

        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'\d+\.\d+\.\d+\.\d+', 'ip_address', text)
        text = re.sub(r'\b\d+\b', 'num', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def get_topic_terms(self, n_terms: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top terms for each LSA component (topic).

        Each component is a linear combination of terms. Terms with high
        absolute weights are most characteristic of that topic.

        Args:
            n_terms: Number of top terms to return per component

        Returns:
            Dictionary mapping component index to list of (term, weight) tuples
        """
        if not self.fitted:
            raise ValueError("LSA analyzer must be fitted first")

        terms = self.tfidf.get_feature_names_out()
        topic_terms = {}

        for i, component in enumerate(self.svd.components_):
            top_indices = np.argsort(np.abs(component))[-n_terms:][::-1]
            topic_terms[i] = [
                (terms[idx], float(component[idx]))
                for idx in top_indices
            ]

        return topic_terms

    def print_topics(self, n_terms: int = 10) -> None:
        """
        Print a formatted summary of all topics.

        Args:
            n_terms: Number of top terms to show per topic
        """
        topic_terms = self.get_topic_terms(n_terms)

        print("\n" + "=" * 60)
        print("LSA TOPIC SUMMARY")
        print("=" * 60)

        for topic_id, terms in topic_terms.items():
            variance = self.svd.explained_variance_ratio_[topic_id]
            print(f"\nTopic {topic_id} (explains {variance:.2%} of variance):")
            print("-" * 40)

            for term, weight in terms:
                direction = "+" if weight > 0 else "-"
                print(f"  {direction} {term}: {abs(weight):.4f}")

    def get_document_topics(
        self,
        texts: List[str],
        top_n: int = 3
    ) -> List[List[Tuple[int, float]]]:
        """
        Get dominant topics for each document.

        Args:
            texts: List of text documents
            top_n: Number of top topics to return per document

        Returns:
            List of lists, each containing (topic_id, score) tuples
        """
        lsa_features = self.transform(texts) if self.fitted else self.fit_transform(texts)

        doc_topics = []
        for doc_vector in lsa_features:
            top_indices = np.argsort(np.abs(doc_vector))[-top_n:][::-1]
            topics = [(int(idx), float(doc_vector[idx])) for idx in top_indices]
            doc_topics.append(topics)

        return doc_topics

    def get_explained_variance(self) -> Dict[str, float]:
        """
        Get explained variance information.

        Returns:
            Dictionary with variance statistics
        """
        if not self.fitted:
            return {}

        return {
            "total_explained": float(self.svd.explained_variance_ratio_.sum()),
            "per_component": self.svd.explained_variance_ratio_.tolist(),
            "cumulative": np.cumsum(self.svd.explained_variance_ratio_).tolist(),
        }

    def get_similar_terms(self, term: str, n_terms: int = 10) -> List[Tuple[str, float]]:
        """
        Find terms similar to a given term in the LSA space.

        Args:
            term: Target term
            n_terms: Number of similar terms to return

        Returns:
            List of (term, similarity) tuples
        """
        if not self.fitted:
            raise ValueError("LSA analyzer must be fitted first")

        vocab = self.tfidf.vocabulary_
        if term not in vocab:
            return []

        term_idx = vocab[term]
        terms = self.tfidf.get_feature_names_out()

        term_vector = self.svd.components_[:, term_idx]

        similarities = []
        for i in range(len(terms)):
            if i == term_idx:
                continue
            other_vector = self.svd.components_[:, i]

            dot_product = np.dot(term_vector, other_vector)
            norm_product = np.linalg.norm(term_vector) * np.linalg.norm(other_vector)
            if norm_product > 0:
                similarity = dot_product / norm_product
                similarities.append((terms[i], float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_terms]

    def get_state_dict(self) -> Dict:
        """
        Get state dictionary for saving the analyzer.

        Returns:
            Dictionary containing analyzer state
        """
        return {
            "config": {
                "n_components": self.config.n_components,
                "tfidf_max_df": self.config.tfidf_max_df,
                "tfidf_min_df": self.config.tfidf_min_df,
                "tfidf_ngram_range": self.config.tfidf_ngram_range,
                "tfidf_max_features": self.config.tfidf_max_features,
            },
            "vocabulary": self.tfidf.vocabulary_ if self.tfidf else None,
            "idf_weights": self.tfidf.idf_.tolist() if self.tfidf else None,
            "svd_components": self.svd.components_.tolist() if self.svd else None,
            "explained_variance": self.get_explained_variance(),
        }
