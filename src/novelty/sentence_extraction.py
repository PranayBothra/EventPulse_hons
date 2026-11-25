"""Sentence extraction and embedding pipeline for EventPulse."""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.novelty.sentence_filtering import SentenceFilter
from src.representation.embedding import BaseEmbeddingProvider, get_embedding_provider
from src.storage.repository import SQLiteRepository


@dataclass
class ExtractedSentence:
    """Represents a clean, validated sentence from an article."""
    sentence_idx: int
    sentence_text: str
    is_headline: bool
    token_count: int
    informativeness_score: float


class SentenceExtractor:
    """Splits article text into clean, information-bearing sentences."""

    # Abbreviations that should not cause sentence splitting
    ABBREVIATIONS = [
        r"U\.S\.", r"U\.K\.", r"E\.U\.", r"U\.N\.", r"D\.C\.",
        r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\.", r"Prof\.", r"Gen\.", r"Gov\.", r"Sen\.", r"Rep\.",
        r"Jan\.", r"Feb\.", r"Mar\.", r"Apr\.", r"Jun\.", r"Jul\.", r"Aug\.", r"Sept?\.", r"Oct\.", r"Nov\.", r"Dec\.",
        r"Inc\.", r"Corp\.", r"Ltd\.", r"Co\.", r"No\.", r"vs\.", r"e\.g\.", r"i\.e\.", r"al\.",
        r"St\.", r"Ave\.", r"Rd\.", r"Blvd\."
    ]

    def __init__(self, sentence_filter: Optional[SentenceFilter] = None):
        self.filter = sentence_filter or SentenceFilter()
        # Compile abbreviation replacement patterns for boundary protection
        escaped_abbr = [f"({abbr})" for abbr in self.ABBREVIATIONS]
        self._abbr_regex = re.compile("|".join(escaped_abbr), re.IGNORECASE)

    def _protect_abbreviations(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Temporarily mask dots in known abbreviations."""
        placeholders: Dict[str, str] = {}
        idx = 0

        def replace_match(match):
            nonlocal idx
            key = f"__ABBR_{idx}__"
            placeholders[key] = match.group(0)
            idx += 1
            return key

        masked = self._abbr_regex.sub(replace_match, text)
        return masked, placeholders

    def _restore_abbreviations(self, text: str, placeholders: Dict[str, str]) -> str:
        """Restore masked abbreviations."""
        restored = text
        for key, orig in placeholders.items():
            restored = restored.replace(key, orig)
        return restored

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using abbreviation-aware boundaries."""
        if not text or not text.strip():
            return []

        cleaned = re.sub(r"\s+", " ", text).strip()
        masked, placeholders = self._protect_abbreviations(cleaned)

        # Split on sentence terminals followed by space and uppercase or end of string
        # Terminals: . ! ? followed by space or quote + space
        pattern = r"(?<=[.!?])[\"\'”’]?\s+(?=[A-Z0-9\"\'“‘])"
        raw_splits = re.split(pattern, masked)

        sentences = []
        for raw in raw_splits:
            restored = self._restore_abbreviations(raw, placeholders)
            cleaned_s = self.filter.clean_sentence(restored)
            if cleaned_s:
                sentences.append(cleaned_s)
        return sentences

    def extract_from_article(
        self,
        headline: str,
        description: Optional[str] = None,
        body: Optional[str] = None,
        known_entities: Optional[List[str]] = None,
    ) -> List[ExtractedSentence]:
        """Extract validated sentences from headline and body text."""
        extracted: List[ExtractedSentence] = []
        sentence_idx = 0

        # 1. Headline is always evaluated as sentence 0
        clean_headline = self.filter.clean_sentence(headline)
        if clean_headline:
            info_score = self.filter.compute_informativeness_score(
                clean_headline, is_headline=True, known_entities=known_entities
            )
            tokens = clean_headline.split()
            extracted.append(
                ExtractedSentence(
                    sentence_idx=sentence_idx,
                    sentence_text=clean_headline,
                    is_headline=True,
                    token_count=len(tokens),
                    informativeness_score=info_score,
                )
            )
            sentence_idx += 1

        # 2. Extract from description and body without duplicating identical text
        clean_desc = (description or "").strip()
        clean_body = (body or "").strip()

        combined_text = ""
        if clean_desc and clean_body:
            if clean_desc == clean_body:
                combined_text = clean_desc
            elif clean_desc in clean_body:
                combined_text = clean_body
            elif clean_body in clean_desc:
                combined_text = clean_desc
            else:
                combined_text = f"{clean_desc} {clean_body}"
        elif clean_desc:
            combined_text = clean_desc
        elif clean_body:
            combined_text = clean_body

        seen_sentences = {clean_headline.lower()}

        if combined_text:
            candidates = self.split_sentences(combined_text)
            for cand in candidates:
                cand_lower = cand.lower()
                # Avoid exact sentence repetition within article
                if cand_lower in seen_sentences:
                    continue
                seen_sentences.add(cand_lower)

                is_valid, _ = self.filter.is_valid_sentence(cand)
                if not is_valid:
                    continue

                info_score = self.filter.compute_informativeness_score(
                    cand, is_headline=False, known_entities=known_entities
                )
                tokens = cand.split()
                extracted.append(
                    ExtractedSentence(
                        sentence_idx=sentence_idx,
                        sentence_text=cand,
                        is_headline=False,
                        token_count=len(tokens),
                        informativeness_score=info_score,
                    )
                )
                sentence_idx += 1

        return extracted


class SentencePipeline:
    """Manages sentence extraction, embedding generation, and database persistence."""

    def __init__(
        self,
        repository: SQLiteRepository,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        extractor: Optional[SentenceExtractor] = None,
    ):
        self.repo = repository
        self.provider = embedding_provider or get_embedding_provider("minilm")
        self.extractor = extractor or SentenceExtractor()

    def process_and_save_article_sentences(
        self,
        article_id: Any,
        headline: str,
        description: Optional[str] = None,
        body: Optional[str] = None,
        known_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract, embed, and persist sentences for an article.

        Returns the list of saved sentence records with database IDs.
        """
        art_id_int = int(article_id)
        # Check if sentences already exist
        existing = self.repo.get_article_sentences(art_id_int)
        if existing:
            return existing

        sentences = self.extractor.extract_from_article(
            headline=headline,
            description=description,
            body=body,
            known_entities=known_entities,
        )

        if not sentences:
            return []

        sentence_dicts = [
            {
                "article_id": art_id_int,
                "sentence_index": s.sentence_idx,
                "sentence_hash": hashlib.sha256(s.sentence_text.encode("utf-8")).hexdigest()[:16],
                "sentence_text": s.sentence_text,
                "is_information_bearing": True,
                "informativeness_score": s.informativeness_score,
                "named_entities": list(known_entities or []),
                "has_numbers": bool(re.search(r"\d", s.sentence_text)),
                "has_dates": bool(SentenceFilter.TEMPORAL_PATTERNS.search(s.sentence_text)),
                "numerical_tokens": re.findall(r"[\$€£]?\d+(?:\.\d+)?%?", s.sentence_text),
                "token_count": s.token_count,
            }
            for s in sentences
        ]

        # 1. Save sentences to SQLite
        self.repo.save_article_sentences(sentence_dicts)
        saved_records = self.repo.get_article_sentences(art_id_int)

        # 2. Compute embeddings in batch
        texts_to_embed = [r["sentence_text"] for r in saved_records]
        embeddings = self.provider.embed_texts(texts_to_embed)

        # 3. Save sentence embeddings
        embedding_pairs = []
        model_id = self.provider.model_id
        for record, emb in zip(saved_records, embeddings):
            emb_blob = np.asarray(emb, dtype=np.float32).tobytes()
            embedding_pairs.append((record["id"], model_id, emb_blob))

        self.repo.batch_save_sentence_embeddings(embedding_pairs)
        return saved_records

    def batch_process_articles(self, articles: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """Process multiple articles efficiently."""
        total_processed = 0
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            for art in batch:
                art_id = art.get("id") or art.get("article_id")
                headline = art.get("headline", "")
                description = art.get("description", "")
                body = art.get("body", "")
                self.process_and_save_article_sentences(
                    article_id=art_id,
                    headline=headline,
                    description=description,
                    body=body,
                )
                total_processed += 1
        return total_processed
