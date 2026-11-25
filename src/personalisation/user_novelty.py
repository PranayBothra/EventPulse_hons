"""User-level information novelty and redundancy evaluation for EventPulse."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.storage.repository import SQLiteRepository


@dataclass
class UserNoveltyScore:
    """Detailed user-level novelty and redundancy evaluation."""
    user_id: str
    article_id: int
    user_novelty_score: float  # [0.0, 1.0] (1.0 = completely novel to user)
    redundancy_penalty: float  # [0.0, 1.0] (1.0 = completely redundant to user)
    consumed_sentence_count: int
    most_similar_consumed_similarity: float


class UserNoveltyEngine:
    """Computes personalized novelty and redundancy based on user consumption history."""

    def __init__(self, repository: SQLiteRepository, model_id: str = "all-MiniLM-L6-v2"):
        self.repo = repository
        self.model_id = model_id

    def compute_user_novelty(
        self,
        user_id: str,
        article_id: int,
        article_sentence_ids: Optional[List[int]] = None,
    ) -> UserNoveltyScore:
        """Evaluate how much novel information article introduces relative to user's consumption.

        Distinct from Event Novelty:
        Even if an article has low event novelty (old facts for the world), it has HIGH
        user novelty if this specific user has never read about it.
        """
        # 1. Fetch user's consumed sentence IDs
        consumed_ids = self.repo.get_user_consumed_sentence_ids(user_id)
        if not consumed_ids:
            # Completely fresh user: everything is 100% novel
            return UserNoveltyScore(
                user_id=user_id,
                article_id=article_id,
                user_novelty_score=1.0,
                redundancy_penalty=0.0,
                consumed_sentence_count=0,
                most_similar_consumed_similarity=0.0,
            )

        # 2. Fetch candidate article sentences
        if article_sentence_ids is None:
            sentences = self.repo.get_article_sentences(article_id)
            if not sentences:
                return UserNoveltyScore(
                    user_id=user_id,
                    article_id=article_id,
                    user_novelty_score=1.0,
                    redundancy_penalty=0.0,
                    consumed_sentence_count=len(consumed_ids),
                    most_similar_consumed_similarity=0.0,
                )
            target_ids = [s["id"] for s in sentences]
        else:
            target_ids = article_sentence_ids
            sentences = [s for s in self.repo.get_article_sentences(article_id) if s["id"] in target_ids]

        # 3. Retrieve embeddings for target sentences and consumed sentences
        # For performance, bound consumed sentences to most recent 200
        recent_consumed_ids = list(consumed_ids)[-200:]
        all_ids = list(set(target_ids + recent_consumed_ids))
        embeddings_map = self.repo.get_sentence_embeddings(all_ids, model_id=self.model_id)

        target_vectors = []
        target_weights = []
        for s in sentences:
            sid = s["id"]
            blob = embeddings_map.get(sid)
            if blob is not None:
                vec = np.frombuffer(blob, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                target_vectors.append(vec)
                is_headline = (s.get("sentence_index", 0) == 0)
                weight = s.get("informativeness_score", 1.0) * (1.5 if is_headline else 1.0)
                target_weights.append(weight)

        if not target_vectors:
            return UserNoveltyScore(
                user_id=user_id,
                article_id=article_id,
                user_novelty_score=1.0,
                redundancy_penalty=0.0,
                consumed_sentence_count=len(consumed_ids),
                most_similar_consumed_similarity=0.0,
            )

        consumed_vectors = []
        for cid in recent_consumed_ids:
            blob = embeddings_map.get(cid)
            if blob is not None:
                vec = np.frombuffer(blob, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                consumed_vectors.append(vec)

        if not consumed_vectors:
            return UserNoveltyScore(
                user_id=user_id,
                article_id=article_id,
                user_novelty_score=1.0,
                redundancy_penalty=0.0,
                consumed_sentence_count=len(consumed_ids),
                most_similar_consumed_similarity=0.0,
            )

        # 4. Compute cosine similarity matrix: (num_targets, num_consumed)
        target_matrix = np.vstack(target_vectors)
        consumed_matrix = np.vstack(consumed_vectors)
        similarity_matrix = np.dot(target_matrix, consumed_matrix.T)

        max_sims_per_sentence = np.max(similarity_matrix, axis=1)
        max_overall_sim = float(np.max(max_sims_per_sentence))

        # Sentence-level user novelty: 1 - max_sim
        sentence_user_novelties = [
            max(0.0, min(1.0, 1.0 - max(0.0, float(sim)))) for sim in max_sims_per_sentence
        ]

        # Weighted aggregate
        sum_weights = sum(target_weights)
        if sum_weights > 0:
            agg_user_novelty = sum(w * n for w, n in zip(target_weights, sentence_user_novelties)) / sum_weights
        else:
            agg_user_novelty = float(np.mean(sentence_user_novelties))

        agg_user_novelty = max(0.0, min(1.0, round(float(agg_user_novelty), 4)))
        redundancy_penalty = round(1.0 - agg_user_novelty, 4)

        return UserNoveltyScore(
            user_id=user_id,
            article_id=article_id,
            user_novelty_score=agg_user_novelty,
            redundancy_penalty=redundancy_penalty,
            consumed_sentence_count=len(consumed_ids),
            most_similar_consumed_similarity=round(max_overall_sim, 4),
        )

    def compute_event_exposure_penalty(
        self, user_id: str, event_id: str, lambda_exposure: float = 0.5
    ) -> float:
        """Compute diminishing returns penalty for repeated exposures to the same event."""
        query = """
        SELECT COUNT(ui.id)
        FROM user_interactions ui
        JOIN event_articles ea ON ui.article_id = ea.article_id
        WHERE ui.user_id = ? AND ea.event_id = ?;
        """
        with self.repo._get_connection() as conn:
            count = conn.execute(query, (user_id, event_id)).fetchone()[0]

        if count == 0:
            return 0.0

        # Exponential saturation penalty: 1 - exp(-lambda * count)
        penalty = 1.0 - np.exp(-lambda_exposure * count)
        return float(round(penalty, 4))
