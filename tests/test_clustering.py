"""Unit tests for temporal event clustering engine, similarity scoring, and lifecycle."""

from typing import Optional
import numpy as np
import pytest

from src.clustering.engine import TemporalEventClusteringEngine
from src.clustering.models import EventCluster
from src.clustering.similarity import MultiSignalSimilarity
from src.config import GlobalConfig
from src.representation.article_representation import ArticleRepresentation
from src.representation.embedding import TfidfEmbeddingProvider
from src.storage.repository import SQLiteRepository


def make_dummy_article(
    article_id: int,
    headline: str,
    description: str,
    published_at: str,
    embedding: np.ndarray,
    entities: set,
    source_id: str = "test_src",
    repo: Optional[SQLiteRepository] = None,
) -> ArticleRepresentation:
    """Helper to construct deterministic ArticleRepresentation fixtures and optionally insert DB record."""
    from src.representation.article_representation import parse_timestamp_to_epoch

    if repo is not None:
        with repo._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO articles (
                    id, canonical_url, url, headline, headline_clean, headline_hash,
                    description, content_clean, source_id, source_name, published_at, retrieved_at, is_sample, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id,
                    f"https://example.com/art/{article_id}",
                    f"https://example.com/art/{article_id}",
                    headline,
                    headline.lower(),
                    f"hash_{article_id}",
                    description,
                    description,
                    source_id,
                    source_id.upper(),
                    published_at,
                    published_at,
                    0,
                    published_at,
                ),
            )
            conn.commit()

    epoch = parse_timestamp_to_epoch(published_at)
    return ArticleRepresentation(
        article_id=article_id,
        headline=headline,
        description=description,
        source_id=source_id,
        source_name=source_id.upper(),
        source_url=f"https://example.com/art/{article_id}",
        published_at=published_at,
        retrieved_at=published_at,
        published_epoch=epoch,
        text_for_embedding=f"{headline}. {description}",
        text_hash=f"hash_{article_id}",
        embedding=embedding,
        entities={"OTHER": entities},
        normalized_entities=entities,
    )


class TestSimilarityScoring:
    def test_same_event_high_similarity(self):
        sim = MultiSignalSimilarity()
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        art1 = make_dummy_article(
            1, "Jensen Huang Reinstated as Nvidia Chief", "Trump intervenes in leadership",
            "2026-11-23T12:00:00Z", v, {"jensen huang", "nvidia", "donald trump"}
        )
        art2 = make_dummy_article(
            2, "Jensen Huang Returns to Nvidia Following Trump Intervention", "Nvidia board reinstates CEO",
            "2026-11-23T13:00:00Z", v, {"jensen huang", "nvidia", "donald trump"}
        )

        cluster = EventCluster(
            id="evt_test", title=art1.headline, summary=art1.description,
            first_seen_at=art1.published_at, last_seen_at=art1.published_at,
            centroid=art1.embedding, entities=set(art1.normalized_entities),
            headline_tokens=sim.tokenize_headline(art1.headline),
        )

        score, breakdown = sim.compute_similarity(art2, cluster)
        assert score > 0.70
        assert breakdown["semantic"] == 1.0
        assert breakdown["entity"] == 1.0
        assert breakdown["temporal"] > 0.95

    def test_different_events_with_shared_entity(self):
        """Verify that sharing an entity (SpaceX) does NOT force match when topic and semantics differ."""
        sim = MultiSignalSimilarity()
        v_grok = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v_military = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Orthogonal vector

        art_grok = make_dummy_article(
            1, "US States Sue SpaceX Over Grok AI Tool", "Child safety lawsuit filed",
            "2026-11-23T10:00:00Z", v_grok, {"spacex", "grok"}
        )
        art_military = make_dummy_article(
            2, "SpaceX Starshield Faces Setbacks in Ukraine Operations", "Defense contract challenges",
            "2026-11-23T11:00:00Z", v_military, {"spacex", "starshield", "ukraine"}
        )

        cluster_grok = EventCluster(
            id="evt_grok", title=art_grok.headline, summary="",
            first_seen_at=art_grok.published_at, last_seen_at=art_grok.published_at,
            centroid=art_grok.embedding, entities=set(art_grok.normalized_entities),
            headline_tokens=sim.tokenize_headline(art_grok.headline),
        )

        score, breakdown = sim.compute_similarity(art_military, cluster_grok)
        # Even with shared entity 'spacex', semantic similarity is 0.0 and headline overlap is low
        assert score < 0.45
        assert breakdown["semantic"] == 0.0

    def test_temporal_compatibility_decay(self):
        sim = MultiSignalSimilarity(temporal_half_life_hours=48.0)
        t_now = "2026-11-24T12:00:00Z"
        from src.representation.article_representation import parse_timestamp_to_epoch

        now_epoch = parse_timestamp_to_epoch(t_now)
        # Exactly 48 hours earlier
        t_48h = "2026-11-22T12:00:00Z"
        compat_48 = sim.compute_temporal_compatibility(now_epoch, t_48h)
        assert pytest.approx(compat_48, rel=1e-2) == 0.50

        # Exactly 96 hours earlier (2 half-lives)
        t_96h = "2026-11-20T12:00:00Z"
        compat_96 = sim.compute_temporal_compatibility(now_epoch, t_96h)
        assert pytest.approx(compat_96, rel=1e-2) == 0.25

    def test_missing_or_invalid_timestamps(self):
        sim = MultiSignalSimilarity()
        from src.representation.article_representation import parse_timestamp_to_epoch

        # Invalid date string falls back to current epoch gracefully
        epoch_fallback = parse_timestamp_to_epoch("not-a-date")
        compat = sim.compute_temporal_compatibility(epoch_fallback, "2026-11-24T12:00:00Z")
        assert 0.0 <= compat <= 1.0


class TestClusteringEngine:
    @pytest.fixture
    def test_engine(self, temp_db: SQLiteRepository, test_config: GlobalConfig):
        provider = TfidfEmbeddingProvider(dimension=64)
        engine = TemporalEventClusteringEngine(
            config=test_config,
            repository=temp_db,
            embedding_provider=provider,
        )
        return engine

    def test_new_event_creation(self, test_engine: TemporalEventClusteringEngine):
        v = np.zeros(64, dtype=np.float32)
        v[0] = 1.0
        art = make_dummy_article(
            1, "Volcanic Eruption in Iceland", "Lava spews near Grindavik",
            "2026-11-24T08:00:00Z", v, {"iceland"}, repo=test_engine.repository
        )

        decision = test_engine.assign_article(art, persist=True)
        assert decision.is_new_event is True
        assert decision.event_id in test_engine.active_events
        assert test_engine.active_events[decision.event_id].article_count == 1

    def test_event_assignment_and_centroid_update(self, test_engine: TemporalEventClusteringEngine):
        v1 = np.zeros(64, dtype=np.float32)
        v1[0] = 1.0
        art1 = make_dummy_article(
            1, "Volcanic Eruption in Iceland Prompts Evacuations", "Grindavik evacuated",
            "2026-11-24T08:00:00Z", v1, {"iceland", "grindavik"}, repo=test_engine.repository
        )

        d1 = test_engine.assign_article(art1, persist=True)
        assert d1.is_new_event is True
        ev_id = d1.event_id

        # Similar article arriving 1 hour later
        v2 = np.zeros(64, dtype=np.float32)
        v2[0] = 0.9
        v2[1] = 0.1
        v2 /= np.linalg.norm(v2)
        art2 = make_dummy_article(
            2, "Iceland Volcano Erupts Near Town of Grindavik", "Lava flows toward infrastructure",
            "2026-11-24T09:00:00Z", v2, {"iceland", "grindavik"}, repo=test_engine.repository
        )

        d2 = test_engine.assign_article(art2, persist=True)
        assert d2.is_new_event is False
        assert d2.event_id == ev_id
        assert test_engine.active_events[ev_id].article_count == 2
        # Centroid should be updated
        c = test_engine.active_events[ev_id].centroid
        assert pytest.approx(np.linalg.norm(c), rel=1e-3) == 1.0

    def test_ambiguous_event_assignment(self, test_engine: TemporalEventClusteringEngine):
        # Create two competing events that are both moderately similar to an incoming article
        v_a = np.zeros(64, dtype=np.float32)
        v_a[0] = 1.0
        art_a = make_dummy_article(
            1, "Naval Drone Attacks Russian Fleet", "Black sea incident",
            "2026-11-24T05:00:00Z", v_a, {"russia", "navy"}, repo=test_engine.repository
        )
        test_engine.assign_article(art_a, persist=True)

        v_b = np.zeros(64, dtype=np.float32)
        v_b[0] = 0.95
        v_b[1] = 0.05
        v_b /= np.linalg.norm(v_b)
        art_b = make_dummy_article(
            2, "Russian Warship Sinks After Drone Strike", "Mediterranean incident",
            "2026-11-24T06:00:00Z", v_b, {"russia", "warship"}, repo=test_engine.repository
        )
        test_engine.assign_article(art_b, persist=True)

        # Incoming article with intermediate representation
        v_c = np.zeros(64, dtype=np.float32)
        v_c[0] = 0.98
        v_c /= np.linalg.norm(v_c)
        art_c = make_dummy_article(
            3, "Ukraine Confirms Drone Strike on Russian Vessel", "Military statement released",
            "2026-11-24T07:00:00Z", v_c, {"russia"}, repo=test_engine.repository
        )

        d3 = test_engine.assign_article(art_c, persist=True)
        # Should assign to one, but identify competing event
        assert d3.is_new_event is False
        if d3.is_ambiguous:
            assert d3.competing_event_id is not None
            assert d3.competing_score is not None

    def test_persistence_and_database_restart(self, temp_db: SQLiteRepository, test_config: GlobalConfig):
        provider = TfidfEmbeddingProvider(dimension=64)

        # Session 1: Create engine and assign article
        engine1 = TemporalEventClusteringEngine(config=test_config, repository=temp_db, embedding_provider=provider)
        v = np.zeros(64, dtype=np.float32)
        v[5] = 1.0
        art = make_dummy_article(
            100, "Major Earthquake Hits Pacific Coast", "Tsunami warning issued",
            "2026-11-24T10:00:00Z", v, {"pacific"}, repo=temp_db
        )
        d = engine1.assign_article(art, persist=True)
        created_id = d.event_id

        # Verify in SQLite directly
        db_ev = temp_db.get_event_by_id(created_id)
        assert db_ev is not None
        assert db_ev["title"] == art.headline

        # Session 2: Instantiate new engine (simulating application restart)
        engine2 = TemporalEventClusteringEngine(config=test_config, repository=temp_db, embedding_provider=provider)
        assert created_id in engine2.active_events
        restarted_ev = engine2.active_events[created_id]
        assert restarted_ev.title == art.headline
        assert restarted_ev.article_count == 1
        assert restarted_ev.centroid is not None
        np.testing.assert_allclose(restarted_ev.centroid, v, atol=1e-4)
