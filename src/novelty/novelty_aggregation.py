"""Article-level novelty aggregation and Event Novelty Engine for EventPulse."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from src.novelty.event_memory import EventMemoryManager, MemorySentence
from src.novelty.novelty_scoring import SentenceNoveltyResult, SentenceNoveltyScorer
from src.novelty.sentence_extraction import SentencePipeline
from src.storage.repository import SQLiteRepository


@dataclass
class ArticleNoveltySummary:
    """Aggregated novelty evaluation for an entire news article within an event."""
    article_id: int
    event_id: str
    event_novelty_score: float
    novel_sentence_count: int
    redundant_sentence_count: int
    insufficient_evidence_count: int
    total_sentence_count: int
    novelty_ratio: float
    sentence_results: List[SentenceNoveltyResult]


class EventNoveltyEngine:
    """Evaluates incremental information novelty of articles within evolving events."""

    def __init__(
        self,
        repository: SQLiteRepository,
        memory_manager: Optional[EventMemoryManager] = None,
        scorer: Optional[SentenceNoveltyScorer] = None,
        pipeline: Optional[SentencePipeline] = None,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        self.repo = repository
        self.model_id = model_id
        self.memory_manager = memory_manager or EventMemoryManager(repository, model_id=model_id)
        self.scorer = scorer or SentenceNoveltyScorer()
        self.pipeline = pipeline or SentencePipeline(repository)

    def evaluate_article_novelty(
        self,
        article_id: int,
        event_id: str,
        published_at: str,
        published_epoch: float,
        persist: bool = True,
    ) -> ArticleNoveltySummary:
        """Evaluate article novelty against chronological event memory prior to published_epoch.

        Strict Temporal Invariant:
        Only historical memory with epoch < published_epoch is referenced.
        """
        # 1. Ensure sentences are extracted and embedded
        sentences = self.repo.get_article_sentences(article_id)
        if not sentences:
            # Look up article details to process sentences
            with self.repo._get_connection() as conn:
                row = conn.execute(
                    "SELECT headline, description, content_clean FROM articles WHERE id = ?",
                    (article_id,),
                ).fetchone()
            if row:
                sentences = self.pipeline.process_and_save_article_sentences(
                    article_id=article_id,
                    headline=row["headline"],
                    description=row["description"],
                    body=row["content_clean"],
                )
            else:
                sentences = []

        if not sentences:
            empty_summary = ArticleNoveltySummary(
                article_id=article_id,
                event_id=event_id,
                event_novelty_score=0.0,
                novel_sentence_count=0,
                redundant_sentence_count=0,
                insufficient_evidence_count=0,
                total_sentence_count=0,
                novelty_ratio=0.0,
                sentence_results=[],
            )
            return empty_summary

        # 2. Fetch sentence embeddings
        sentence_ids = [s["id"] for s in sentences]
        embeddings_map = self.repo.get_sentence_embeddings(sentence_ids, model_id=self.model_id)

        # 3. Retrieve chronological prior memory H(e, t)
        prior_memory = self.memory_manager.get_event_memory(event_id, before_epoch=published_epoch)

        # 4. Evaluate each sentence against memory (and intra-article preceding sentences)
        current_memory = list(prior_memory)
        sentence_results: List[SentenceNoveltyResult] = []
        memory_to_commit: List[Dict[str, Any]] = []

        weights: List[float] = []
        novelties: List[float] = []

        novel_count = 0
        redundant_count = 0
        insufficient_count = 0

        for s in sentences:
            sid = s["id"]
            blob = embeddings_map.get(sid)
            if blob is not None:
                vec = np.frombuffer(blob, dtype=np.float32)
            else:
                vec = np.zeros(384, dtype=np.float32)

            res = self.scorer.score_sentence(
                sentence_id=sid,
                article_id=article_id,
                sentence_text=s["sentence_text"],
                sentence_index=s["sentence_index"],
                embedding=vec,
                informativeness_score=s["informativeness_score"],
                memory=current_memory,
            )
            sentence_results.append(res)

            # Weighting: headline gets 1.5x prominence, scaled by informativeness
            is_headline = (s["sentence_index"] == 0)
            weight = s["informativeness_score"] * (1.5 if is_headline else 1.0)
            weights.append(weight)
            novelties.append(res.novelty_score)

            if res.evidence_status == "POTENTIALLY_NEW":
                novel_count += 1
            elif res.evidence_status == "REPEATED_INFORMATION":
                redundant_count += 1
            else:
                insufficient_count += 1

            # Prepare memory record to append
            memory_to_commit.append({
                "event_id": event_id,
                "sentence_id": sid,
                "article_id": article_id,
                "published_at": published_at,
                "published_epoch": published_epoch,
                "novelty_score": res.novelty_score,
                "evidence_status": res.evidence_status,
                "most_similar_sentence_id": res.most_similar_sentence_id,
                "highest_similarity": res.highest_similarity,
            })

            # Append current sentence to active memory for succeeding sentences of this article
            current_memory.append(
                MemorySentence(
                    sentence_id=sid,
                    article_id=article_id,
                    published_at=published_at,
                    published_epoch=published_epoch,
                    sentence_text=s["sentence_text"],
                    informativeness_score=s["informativeness_score"],
                    embedding=vec,
                    novelty_score=res.novelty_score,
                    evidence_status=res.evidence_status,
                    named_entities=s["named_entities"],
                    has_numbers=s["has_numbers"],
                    has_dates=s["has_dates"],
                )
            )

        # 5. Compute article-level aggregate novelty
        total_sentences = len(sentences)
        sum_weights = sum(weights)
        if sum_weights > 0:
            agg_novelty = sum(w * n for w, n in zip(weights, novelties)) / sum_weights
        else:
            agg_novelty = float(np.mean(novelties)) if novelties else 0.0

        novelty_ratio = (novel_count / total_sentences) if total_sentences > 0 else 0.0

        summary = ArticleNoveltySummary(
            article_id=article_id,
            event_id=event_id,
            event_novelty_score=round(float(agg_novelty), 4),
            novel_sentence_count=novel_count,
            redundant_sentence_count=redundant_count,
            insufficient_evidence_count=insufficient_count,
            total_sentence_count=total_sentences,
            novelty_ratio=round(float(novelty_ratio), 4),
            sentence_results=sentence_results,
        )

        # 6. Persist to database if requested
        if persist:
            self.memory_manager.commit_sentences_to_memory(event_id, memory_to_commit)
            self.repo.save_article_novelty({
                "article_id": article_id,
                "event_id": event_id,
                "event_novelty_score": summary.event_novelty_score,
                "novel_sentence_count": summary.novel_sentence_count,
                "redundant_sentence_count": summary.redundant_sentence_count,
                "total_sentence_count": summary.total_sentence_count,
                "novelty_ratio": summary.novelty_ratio,
            })

        return summary
