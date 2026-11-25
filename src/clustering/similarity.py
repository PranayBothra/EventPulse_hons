"""Multi-signal similarity scoring for temporal event clustering."""

import math
from datetime import datetime, timezone
from typing import Dict, Optional, Set, Tuple
import numpy as np

from src.clustering.models import EventCluster
from src.preprocessing.deduplicator import STOPWORDS, normalize_headline_for_comparison
from src.representation.article_representation import ArticleRepresentation, parse_timestamp_to_epoch


class MultiSignalSimilarity:
    """Computes calibrated multi-signal similarity between an article and an event cluster."""

    def __init__(
        self,
        weight_semantic: float = 0.45,
        weight_entity: float = 0.25,
        weight_temporal: float = 0.20,
        weight_headline: float = 0.10,
        temporal_half_life_hours: float = 48.0,
    ):
        # Normalize weights so they sum to 1.0
        total_w = weight_semantic + weight_entity + weight_temporal + weight_headline
        if total_w <= 0:
            total_w = 1.0
        self.w_semantic = weight_semantic / total_w
        self.w_entity = weight_entity / total_w
        self.w_temporal = weight_temporal / total_w
        self.w_headline = weight_headline / total_w

        self.half_life_hours = max(1.0, temporal_half_life_hours)
        self.decay_lambda = math.log(2.0) / self.half_life_hours

    def tokenize_headline(self, headline: str) -> Set[str]:
        """Extract clean non-stopword tokens from a headline."""
        norm = normalize_headline_for_comparison(headline)
        return {w for w in norm.split() if w not in STOPWORDS and len(w) > 2}

    def compute_semantic_similarity(
        self, article_emb: np.ndarray, event_centroid: Optional[np.ndarray]
    ) -> float:
        """Compute cosine similarity between normalized vectors."""
        if event_centroid is None or article_emb is None:
            return 0.0

        dot = float(np.dot(article_emb, event_centroid))
        # Clip to [0.0, 1.0]
        return max(0.0, min(1.0, dot))

    def compute_entity_similarity(
        self, article_entities: Set[str], event_entities: Set[str]
    ) -> float:
        """Compute Jaccard similarity over normalized entity sets."""
        if not article_entities and not event_entities:
            return 0.5  # Neutral fallback when neither possesses recognized entities

        union = len(article_entities | event_entities)
        if union == 0:
            return 0.5

        intersection = len(article_entities & event_entities)
        return intersection / union

    def compute_temporal_compatibility(
        self, article_epoch: float, event_last_seen_at: str
    ) -> float:
        """Compute exponential temporal decay compatibility."""
        event_epoch = parse_timestamp_to_epoch(event_last_seen_at)
        diff_hours = abs(article_epoch - event_epoch) / 3600.0

        score = math.exp(-self.decay_lambda * diff_hours)
        return max(0.0, min(1.0, score))

    def compute_headline_similarity(
        self, article_tokens: Set[str], event_tokens: Set[str]
    ) -> float:
        """Compute token-level Jaccard similarity across headlines."""
        if not article_tokens or not event_tokens:
            return 0.0

        intersection = len(article_tokens & event_tokens)
        union = len(article_tokens | event_tokens)
        if union == 0:
            return 0.0

        return intersection / union

    def compute_similarity(
        self,
        article: ArticleRepresentation,
        event: EventCluster,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate the full composite multi-signal similarity score.

        Returns:
            Tuple of (composite_score, score_breakdown_dict).
        """
        sem_sim = self.compute_semantic_similarity(
            article.embedding, event.centroid
        )
        ent_sim = self.compute_entity_similarity(
            article.normalized_entities, event.entities
        )
        temp_sim = self.compute_temporal_compatibility(
            article.published_epoch, event.last_seen_at
        )

        article_tokens = self.tokenize_headline(article.headline)
        head_sim = self.compute_headline_similarity(
            article_tokens, event.headline_tokens
        )

        composite = (
            self.w_semantic * sem_sim
            + self.w_entity * ent_sim
            + self.w_temporal * temp_sim
            + self.w_headline * head_sim
        )
        composite = max(0.0, min(1.0, composite))

        breakdown = {
            "semantic": sem_sim,
            "entity": ent_sim,
            "temporal": temp_sim,
            "headline": head_sim,
            "composite": composite,
        }
        return composite, breakdown

    def compute_article_pair_similarity(
        self,
        a1: ArticleRepresentation,
        a2: ArticleRepresentation,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate composite multi-signal similarity directly between two articles."""
        sem_sim = self.compute_semantic_similarity(a1.embedding, a2.embedding)
        ent_sim = self.compute_entity_similarity(
            a1.normalized_entities, a2.normalized_entities
        )
        temp_sim = self.compute_temporal_compatibility(
            a1.published_epoch, a2.published_at
        )
        t1 = self.tokenize_headline(a1.headline)
        t2 = self.tokenize_headline(a2.headline)
        head_sim = self.compute_headline_similarity(t1, t2)

        composite = (
            self.w_semantic * sem_sim
            + self.w_entity * ent_sim
            + self.w_temporal * temp_sim
            + self.w_headline * head_sim
        )
        composite = max(0.0, min(1.0, composite))
        breakdown = {
            "semantic": sem_sim,
            "entity": ent_sim,
            "temporal": temp_sim,
            "headline": head_sim,
            "composite": composite,
        }
        return composite, breakdown
