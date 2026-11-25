"""Mathematical scoring and evidence classification for sentence-level novelty."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from src.novelty.event_memory import MemorySentence


@dataclass
class SentenceNoveltyResult:
    """Detailed novelty scoring result for a single sentence."""
    sentence_id: int
    article_id: int
    sentence_text: str
    sentence_index: int
    highest_similarity: float
    most_similar_sentence_id: Optional[int]
    most_similar_text: Optional[str]
    novelty_score: float
    informativeness_score: float
    evidence_status: str  # POTENTIALLY_NEW, REPEATED_INFORMATION, INSUFFICIENT_EVIDENCE


class SentenceNoveltyScorer:
    """Scores sentence novelty against chronological event memory."""

    def __init__(
        self,
        redundancy_threshold: float = 0.75,
        novelty_threshold: float = 0.65,
        min_informative_threshold: float = 0.35,
    ):
        self.redundancy_threshold = redundancy_threshold
        self.novelty_threshold = novelty_threshold
        self.min_informative_threshold = min_informative_threshold

    def score_sentence(
        self,
        sentence_id: int,
        article_id: int,
        sentence_text: str,
        sentence_index: int,
        embedding: np.ndarray,
        informativeness_score: float,
        memory: List[MemorySentence],
    ) -> SentenceNoveltyResult:
        """Compute cosine similarity against memory, raw novelty, and evidence label."""
        # Ensure unit normalization of candidate vector
        norm = np.linalg.norm(embedding)
        cand_vec = embedding / norm if norm > 0 else embedding

        if not memory:
            # First information in event
            highest_sim = 0.0
            most_sim_id = None
            most_sim_text = None
        else:
            # Vectorized dot product against all memory vectors
            mem_matrix = np.vstack([m.embedding for m in memory])
            sims = np.dot(mem_matrix, cand_vec)
            best_idx = int(np.argmax(sims))
            highest_sim = float(sims[best_idx])
            most_sim_id = memory[best_idx].sentence_id
            most_sim_text = memory[best_idx].sentence_text

        # Novelty formula: (1 - max_sim) * informativeness
        clamped_sim = max(0.0, min(1.0, highest_sim))
        raw_novelty = (1.0 - clamped_sim) * informativeness_score
        clamped_novelty = max(0.0, min(1.0, round(float(raw_novelty), 4)))

        # Evidence classification
        if highest_sim >= self.redundancy_threshold:
            evidence_status = "REPEATED_INFORMATION"
        elif highest_sim < self.novelty_threshold and informativeness_score >= self.min_informative_threshold:
            evidence_status = "POTENTIALLY_NEW"
        else:
            evidence_status = "INSUFFICIENT_EVIDENCE"

        return SentenceNoveltyResult(
            sentence_id=sentence_id,
            article_id=article_id,
            sentence_text=sentence_text,
            sentence_index=sentence_index,
            highest_similarity=round(clamped_sim, 4),
            most_similar_sentence_id=most_sim_id,
            most_similar_text=most_sim_text,
            novelty_score=clamped_novelty,
            informativeness_score=round(informativeness_score, 4),
            evidence_status=evidence_status,
        )
