"""Unit tests for sentence extraction, filtering, and embedding pipeline."""

import tempfile
import pytest
import numpy as np

from src.novelty.sentence_filtering import SentenceFilter
from src.novelty.sentence_extraction import ExtractedSentence, SentenceExtractor, SentencePipeline
from src.representation.embedding import TfidfEmbeddingProvider
from src.storage.repository import SQLiteRepository


class TestSentenceFilter:
    """Test sentence cleaning, boilerplate rejection, and informativeness scoring."""

    def test_clean_sentence(self):
        sf = SentenceFilter()
        raw = '  "The prime minister announced the new budget."  '
        cleaned = sf.clean_sentence(raw)
        assert cleaned == "The prime minister announced the new budget."

    def test_filter_photo_credit(self):
        sf = SentenceFilter()
        assert not sf.is_valid_sentence("Photo: Getty Images / John Doe")[0]
        assert not sf.is_valid_sentence("Image copyright Reuters/2026")[0]

    def test_filter_newsletter_subscription(self):
        sf = SentenceFilter()
        assert not sf.is_valid_sentence("Sign up for our morning newsletter to stay informed.")[0]
        assert not sf.is_valid_sentence("Click here to subscribe to the daily update.")[0]

    def test_filter_short_and_few_words(self):
        sf = SentenceFilter()
        assert not sf.is_valid_sentence("Hello.")[0]
        assert not sf.is_valid_sentence("Short sentence")[0]

    def test_valid_news_sentence(self):
        sf = SentenceFilter()
        valid, reason = sf.is_valid_sentence(
            "The European Central Bank raised interest rates by 25 basis points on Thursday."
        )
        assert valid
        assert reason == "VALID"

    def test_informativeness_scoring_signals(self):
        sf = SentenceFilter()
        # High factual density: Numbers, dates, entities, action verb
        high_info = sf.compute_informativeness_score(
            "Apple Inc. reported $94.8 billion in revenue on Tuesday, exceeding Wall Street expectations.",
            is_headline=True
        )
        # Low factual density: generic fluff
        low_info = sf.compute_informativeness_score(
            "Needless to say, time will tell what happens next in other news.",
            is_headline=False
        )
        assert high_info > 0.60
        assert low_info < 0.30
        assert high_info > low_info


class TestSentenceExtractor:
    """Test abbreviation-aware sentence splitting and article extraction."""

    def test_abbreviation_preservation(self):
        extractor = SentenceExtractor()
        text = "U.S. officials met with Dr. Smith in Washington D.C. on Jan. 15. The talks focused on trade."
        sentences = extractor.split_sentences(text)
        assert len(sentences) == 2
        assert "U.S. officials met with Dr. Smith in Washington D.C. on Jan. 15." in sentences[0]
        assert "The talks focused on trade." in sentences[1]

    def test_extract_from_article_headline_and_body(self):
        extractor = SentenceExtractor()
        headline = "NASA launches Artemis II lunar mission"
        description = "Four astronauts began their historic journey around the Moon on Monday. Ground control confirmed all systems nominal."
        
        extracted = extractor.extract_from_article(headline=headline, description=description)
        assert len(extracted) == 3
        # Sentence 0 is headline
        assert extracted[0].is_headline is True
        assert extracted[0].sentence_idx == 0
        assert extracted[0].sentence_text == headline
        
        # Sentences 1 and 2
        assert extracted[1].is_headline is False
        assert extracted[1].sentence_idx == 1
        assert "Four astronauts" in extracted[1].sentence_text


class TestSentencePipeline:
    """Test end-to-end sentence extraction and database persistence."""

    @pytest.fixture
    def test_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            repo = SQLiteRepository(tmp.name)
            repo.initialize_schema()
            return repo

    def test_pipeline_process_and_save(self, test_db):
        provider = TfidfEmbeddingProvider(dimension=64)
        pipeline = SentencePipeline(
            repository=test_db,
            embedding_provider=provider,
            extractor=SentenceExtractor()
        )

        # Create dummy feed and article in test_db
        test_db.upsert_feed("feed_1", "Test Feed", "https://example.com/rss")
        test_db.insert_article({
            "canonical_url": "https://example.com/art_101",
            "url": "https://example.com/art_101",
            "headline": "Tech Giants Agree on AI Safety Standards",
            "headline_clean": "tech giants agree on ai safety standards",
            "headline_hash": "hash_art_101",
            "description": "Representatives from major tech firms signed a voluntary pact in Geneva today. The pact addresses frontier model risks.",
            "content_clean": "Representatives from major tech firms signed a voluntary pact in Geneva today. The pact addresses frontier model risks.",
            "source_id": "feed_1",
            "source_name": "TechNews",
            "published_at": "2026-03-01T10:00:00Z",
            "is_sample": False
        })
        art = test_db.get_article_by_canonical_url("https://example.com/art_101")
        art_id = str(art["id"])

        saved_records = pipeline.process_and_save_article_sentences(
            article_id=art_id,
            headline="Tech Giants Agree on AI Safety Standards",
            description="Representatives from major tech firms signed a voluntary pact in Geneva today. The pact addresses frontier model risks."
        )

        assert len(saved_records) == 3
        assert saved_records[0]["sentence_index"] == 0
        
        # Verify sentences persisted
        db_sentences = test_db.get_article_sentences(art_id)
        assert len(db_sentences) == 3

        # Verify embeddings persisted
        sentence_ids = [r["id"] for r in db_sentences]
        embeddings = test_db.get_sentence_embeddings(sentence_ids, model_id=provider.model_id)
        assert len(embeddings) == 3
        for sid in sentence_ids:
            assert sid in embeddings
            vec = np.frombuffer(embeddings[sid], dtype=np.float32)
            assert isinstance(vec, np.ndarray)
            assert len(vec) == 64
