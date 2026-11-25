"""Threshold sensitivity analysis for temporal event clustering."""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from src.clustering.models import EventCluster
from src.clustering.similarity import MultiSignalSimilarity
from src.config import GlobalConfig, load_config
from src.representation.article_representation import (
    ArticleRepresentation,
    build_article_representation,
    parse_timestamp_to_epoch,
)
from src.representation.cache import EmbeddingCacheManager
from src.representation.embedding import BaseEmbeddingProvider
from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.sensitivity")


def run_threshold_sensitivity(
    repository: SQLiteRepository,
    embedding_provider: BaseEmbeddingProvider,
    thresholds: Optional[List[float]] = None,
    is_sample: Optional[bool] = False,
) -> List[Dict[str, Any]]:
    """Simulate clustering across multiple match thresholds to quantify sensitivity.

    Args:
        repository: SQLiteRepository instance.
        embedding_provider: Embedding model provider.
        thresholds: List of float thresholds to test. Defaults to [0.38, 0.42, 0.48, 0.52, 0.58, 0.65].
        is_sample: Process only live articles if False.

    Returns:
        List of result dictionaries per threshold.
    """
    if thresholds is None:
        thresholds = [0.38, 0.42, 0.45, 0.48, 0.52, 0.55, 0.60, 0.65]

    cache_manager = EmbeddingCacheManager(repository)
    raw_articles = repository.get_articles(limit=10000, is_sample=is_sample)
    raw_articles = list(reversed(raw_articles))  # Chronological order

    # Pre-build all article representations in memory to test clustering rapidly
    reps: List[ArticleRepresentation] = [
        build_article_representation(
            article_dict=a,
            embedding_provider=embedding_provider,
            cache_manager=cache_manager,
        )
        for a in raw_articles
    ]

    sim_eval = MultiSignalSimilarity()
    results: List[Dict[str, Any]] = []

    for theta in thresholds:
        # Simulate online clustering for this threshold without touching persistent DB
        events: Dict[str, EventCluster] = {}
        ambiguous_count = 0
        assigned_to_existing = 0

        for rep in reps:
            article_tokens = sim_eval.tokenize_headline(rep.headline)

            # Filter candidates within 168 hours
            candidates = []
            for ev in events.values():
                ev_epoch = parse_timestamp_to_epoch(ev.last_seen_at)
                if abs(rep.published_epoch - ev_epoch) / 3600.0 <= 168.0:
                    candidates.append(ev)

            scored = []
            for ev in candidates:
                s, _ = sim_eval.compute_similarity(rep, ev)
                scored.append((ev, s))

            scored.sort(key=lambda x: x[1], reverse=True)

            if scored and scored[0][1] >= theta:
                best_ev, best_score = scored[0]
                if len(scored) > 1:
                    second_score = scored[1][1]
                    if (best_score - second_score < 0.05) and second_score >= (theta - 0.06):
                        ambiguous_count += 1

                best_ev.add_article(rep, article_tokens)
                assigned_to_existing += 1
            else:
                # Create new cluster
                ev_id = f"sim_{len(events)}_{rep.article_id}"
                new_ev = EventCluster(
                    id=ev_id,
                    title=rep.headline,
                    summary=rep.description,
                    first_seen_at=rep.published_at,
                    last_seen_at=rep.published_at,
                    article_ids=[rep.article_id],
                    article_count=1,
                    centroid=rep.embedding.copy().astype(np.float32),
                    entities=set(rep.normalized_entities),
                    headline_tokens=set(article_tokens),
                )
                events[ev_id] = new_ev

        counts = [ev.article_count for ev in events.values()]
        singletons = sum(1 for c in counts if c == 1)
        multi_clusters = sum(1 for c in counts if c > 1)
        max_cluster = max(counts) if counts else 0
        avg_cluster_size = len(reps) / len(events) if events else 0.0

        results.append(
            {
                "threshold": theta,
                "total_articles": len(reps),
                "total_events": len(events),
                "multi_article_events": multi_clusters,
                "singletons": singletons,
                "max_cluster_size": max_cluster,
                "assigned_to_existing": assigned_to_existing,
                "ambiguous_count": ambiguous_count,
                "avg_cluster_size": round(avg_cluster_size, 3),
            }
        )

    return results
