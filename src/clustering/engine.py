"""Temporal event clustering engine orchestrating candidate search, assignment, and lifecycle management."""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.clustering.models import AssignmentDecision, EventCluster
from src.clustering.similarity import MultiSignalSimilarity
from src.config import GlobalConfig, load_config
from src.entities.extractor import NamedEntityExtractor
from src.representation.article_representation import (
    ArticleRepresentation,
    build_article_representation,
    parse_timestamp_to_epoch,
)
from src.representation.cache import EmbeddingCacheManager
from src.representation.embedding import BaseEmbeddingProvider, get_embedding_provider
from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.clustering")


class TemporalEventClusteringEngine:
    """Online, incremental temporal event clustering engine."""

    def __init__(
        self,
        config: Optional[GlobalConfig] = None,
        repository: Optional[SQLiteRepository] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        entity_extractor: Optional[NamedEntityExtractor] = None,
    ):
        self.config = config or load_config()
        self.repository = repository or SQLiteRepository(
            db_path=self.config.database.path,
            timeout=self.config.database.timeout_seconds,
        )

        c_cfg = self.config.clustering
        self.match_threshold = c_cfg.match_threshold
        self.ambiguity_delta = c_cfg.ambiguity_delta
        self.ambiguity_floor = c_cfg.ambiguity_floor
        self.max_temporal_window_hours = c_cfg.max_temporal_window_hours

        self.similarity_evaluator = MultiSignalSimilarity(
            weight_semantic=c_cfg.weight_semantic,
            weight_entity=c_cfg.weight_entity,
            weight_temporal=c_cfg.weight_temporal,
            weight_headline=c_cfg.weight_headline,
            temporal_half_life_hours=c_cfg.temporal_half_life_hours,
        )

        e_cfg = self.config.embedding
        self.embedding_provider = embedding_provider or get_embedding_provider(
            provider_type=e_cfg.provider,
            model_name=e_cfg.model_name,
        )
        self.cache_manager = EmbeddingCacheManager(self.repository)
        self.entity_extractor = entity_extractor or NamedEntityExtractor(use_spacy=True)

        # In-memory index of active event clusters
        self.active_events: Dict[str, EventCluster] = {}
        self._load_existing_events_from_db()

    def _load_existing_events_from_db(self) -> None:
        """Hydrate in-memory event index from persistent SQLite storage on startup."""
        db_events = self.repository.get_all_events(status=None, limit=500)
        for ev in db_events:
            event_id = ev["id"]
            centroid = None
            if ev.get("centroid_blob"):
                centroid = np.frombuffer(ev["centroid_blob"], dtype=np.float32).copy()

            # Load member articles
            member_rows = self.repository.get_event_articles(event_id)
            article_ids = [r["id"] for r in member_rows]

            # Reconstruct exact embedding sum and centroid if cached vectors exist
            embedding_sum = None
            if article_ids:
                emb_map = self.repository.get_embeddings_by_article_ids(
                    article_ids, model_id=self.embedding_provider.model_id
                )
                if emb_map:
                    vecs = [
                        np.frombuffer(emb_map[aid], dtype=np.float32)
                        for aid in article_ids
                        if aid in emb_map
                    ]
                    if vecs:
                        embedding_sum = np.sum(vecs, axis=0).astype(np.float32)
                        norm = float(np.linalg.norm(embedding_sum))
                        if norm > 1e-12:
                            centroid = (embedding_sum / norm).astype(np.float32)

            if embedding_sum is None and centroid is not None and len(article_ids) > 0:
                embedding_sum = centroid * float(len(article_ids))

            # Reconstruct headline tokens
            tokens: Set[str] = set()
            for r in member_rows:
                tokens.update(
                    self.similarity_evaluator.tokenize_headline(r["headline"])
                )

            cluster = EventCluster(
                id=event_id,
                title=ev["title"],
                summary=ev.get("summary", ""),
                first_seen_at=ev["first_seen_at"],
                last_seen_at=ev["last_seen_at"],
                article_ids=article_ids,
                article_count=len(article_ids),
                centroid=centroid,
                embedding_sum=embedding_sum,
                entities=set(ev.get("key_entities", [])),
                headline_tokens=tokens,
                status=ev.get("status", "ACTIVE"),
            )
            self.active_events[event_id] = cluster

        logger.info(
            "Hydrated %d existing events from database into active clustering index.",
            len(self.active_events),
        )

    def _generate_event_id(self, article: ArticleRepresentation) -> str:
        """Create a clean, human-readable yet unique stable event ID."""
        date_prefix = (
            article.published_at[:10].replace("-", "")
            if article.published_at
            else datetime.now(timezone.utc).strftime("%Y%m%d")
        )

        # Extract leading slug from key entity or headline
        slug = ""
        if article.normalized_entities:
            top_ent = sorted(list(article.normalized_entities), key=len)[-1]
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", top_ent).strip("_")

        if not slug:
            clean_hl = re.sub(r"[^a-zA-Z0-9]+", "_", article.headline.lower()).strip("_")
            slug = clean_hl[:24]

        short_uid = uuid.uuid4().hex[:6]
        return f"evt_{date_prefix}_{slug}_{short_uid}".lower()

    def assign_article(
        self, article: ArticleRepresentation, persist: bool = True
    ) -> AssignmentDecision:
        """Assign an incoming article to an existing event or create a new event.

        Args:
            article: Fully populated ArticleRepresentation.
            persist: If True, immediately writes assignment to SQLite.

        Returns:
            AssignmentDecision detailing matched event, score, and ambiguity status.
        """
        article_tokens = self.similarity_evaluator.tokenize_headline(article.headline)

        # 1. Filter candidate events within the temporal window
        candidate_events = []
        for event in self.active_events.values():
            event_epoch = parse_timestamp_to_epoch(event.last_seen_at)
            diff_hours = abs(article.published_epoch - event_epoch) / 3600.0
            if diff_hours <= self.max_temporal_window_hours:
                candidate_events.append(event)

        # 2. Score candidate events
        scored_candidates: List[Tuple[EventCluster, float, Dict[str, float]]] = []
        for event in candidate_events:
            score, breakdown = self.similarity_evaluator.compute_similarity(
                article, event
            )
            scored_candidates.append((event, score, breakdown))

        # Sort descending by composite score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. Decision logic
        if scored_candidates and scored_candidates[0][1] >= self.match_threshold:
            best_event, best_score, best_breakdown = scored_candidates[0]

            # Ambiguity check
            is_ambiguous = False
            competing_id = None
            competing_score = None

            if len(scored_candidates) > 1:
                second_event, second_score, _ = scored_candidates[1]
                if (
                    best_score - second_score < self.ambiguity_delta
                    and second_score >= self.ambiguity_floor
                ):
                    is_ambiguous = True
                    competing_id = second_event.id
                    competing_score = second_score
                    logger.info(
                        "Ambiguous event match for article %d between '%s' (%.3f) and '%s' (%.3f)",
                        article.article_id,
                        best_event.id,
                        best_score,
                        competing_id,
                        competing_score,
                    )

            # Update event cluster
            best_event.add_article(article, article_tokens)

            # Persist update
            if persist:
                centroid_blob = (
                    best_event.centroid.astype(np.float32).tobytes()
                    if best_event.centroid is not None
                    else None
                )
                self.repository.update_event(
                    event_id=best_event.id,
                    last_seen_at=best_event.last_seen_at,
                    article_count=best_event.article_count,
                    centroid_blob=centroid_blob,
                    key_entities=list(best_event.entities),
                    status=best_event.status,
                )
                self.repository.assign_article_to_event(
                    event_id=best_event.id,
                    article_id=article.article_id,
                    similarity_score=best_score,
                    is_ambiguous=is_ambiguous,
                )

            return AssignmentDecision(
                article_id=article.article_id,
                event_id=best_event.id,
                is_new_event=False,
                similarity_score=best_score,
                is_ambiguous=is_ambiguous,
                competing_event_id=competing_id,
                competing_score=competing_score,
                score_breakdown=best_breakdown,
            )

        else:
            # Create a new event entity
            event_id = self._generate_event_id(article)
            centroid = article.embedding.copy().astype(np.float32)
            new_event = EventCluster(
                id=event_id,
                title=article.headline,
                summary=article.description,
                first_seen_at=article.published_at,
                last_seen_at=article.published_at,
                article_ids=[article.article_id],
                article_count=1,
                centroid=centroid,
                embedding_sum=centroid.copy(),
                entities=set(article.normalized_entities),
                headline_tokens=set(article_tokens),
                status="EMERGING",
            )
            self.active_events[event_id] = new_event

            if persist:
                centroid_blob = centroid.tobytes()
                self.repository.create_event(
                    event_id=event_id,
                    title=new_event.title,
                    summary=new_event.summary,
                    first_seen_at=new_event.first_seen_at,
                    last_seen_at=new_event.last_seen_at,
                    centroid_blob=centroid_blob,
                    key_entities=list(new_event.entities),
                    status="EMERGING",
                )
                self.repository.assign_article_to_event(
                    event_id=event_id,
                    article_id=article.article_id,
                    similarity_score=1.0,
                    is_ambiguous=False,
                )

            return AssignmentDecision(
                article_id=article.article_id,
                event_id=event_id,
                is_new_event=True,
                similarity_score=1.0,
                is_ambiguous=False,
                score_breakdown={"composite": 1.0, "is_initiator": 1.0},
            )

    def cluster_unassigned_articles(
        self, is_sample: Optional[bool] = None
    ) -> List[AssignmentDecision]:
        """Fetch all articles from database in chronological order and perform clustering.

        Args:
            is_sample: If False, processes only live articles. If True, only sample data.
                       If None, processes all articles.
        """
        # Fetch articles ordered chronologically by publication time
        articles = self.repository.get_articles(
            limit=50000,
            offset=0,
            is_sample=is_sample,
        )
        # Reverse to chronological order (get_articles orders DESC)
        articles = list(reversed(articles))

        logger.info("Clustering %d articles in chronological order...", len(articles))
        decisions: List[AssignmentDecision] = []

        # Single query to get all already-assigned article IDs
        with self.repository._get_connection() as conn:
            assigned_ids = {
                r[0]
                for r in conn.execute("SELECT article_id FROM event_articles;").fetchall()
            }

        for art in articles:
            # Check if article is already assigned
            if art["id"] in assigned_ids:
                continue

            rep = build_article_representation(
                article_dict=art,
                embedding_provider=self.embedding_provider,
                cache_manager=self.cache_manager,
                entity_extractor=self.entity_extractor,
                headline_weight=self.config.embedding.headline_weight,
            )
            decision = self.assign_article(rep, persist=True)
            decisions.append(decision)
            assigned_ids.add(art["id"])

        logger.info(
            "Clustering complete. %d articles processed into %d active events.",
            len(decisions),
            len(self.active_events),
        )
        return decisions

    def rebuild_all_event_centroids(self) -> int:
        """Recompute exact unnormalized sum and unit centroid for all events from cached embeddings.

        Returns:
            Number of events whose centroids were safely reconstructed and persisted.
        """
        all_events = self.repository.get_all_events(status=None, limit=10000)
        rebuilt = 0
        for ev in all_events:
            event_id = ev["id"]
            member_rows = self.repository.get_event_articles(event_id)
            article_ids = [r["id"] for r in member_rows]
            if not article_ids:
                continue

            emb_map = self.repository.get_embeddings_by_article_ids(
                article_ids, model_id=self.embedding_provider.model_id
            )
            vecs = [
                np.frombuffer(emb_map[aid], dtype=np.float32)
                for aid in article_ids
                if aid in emb_map
            ]
            if not vecs:
                continue

            sum_vec = np.sum(vecs, axis=0).astype(np.float32)
            norm = float(np.linalg.norm(sum_vec))
            centroid = (
                (sum_vec / norm).astype(np.float32)
                if norm > 1e-12
                else np.zeros_like(sum_vec, dtype=np.float32)
            )

            # Update database
            self.repository.update_event(
                event_id=event_id,
                last_seen_at=ev["last_seen_at"],
                article_count=len(article_ids),
                centroid_blob=centroid.tobytes(),
                key_entities=ev.get("key_entities", []),
                status=ev.get("status", "ACTIVE"),
            )

            if event_id in self.active_events:
                self.active_events[event_id].embedding_sum = sum_vec
                self.active_events[event_id].centroid = centroid

            rebuilt += 1

        logger.info("Rebuilt centroids for %d events using exact unnormalized sum.", rebuilt)
        return rebuilt
