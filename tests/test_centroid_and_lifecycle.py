"""Unit tests verifying mathematical correctness of centroid updates and lifecycle state transitions."""

import numpy as np
import pytest

from src.clustering.models import EventCluster
from src.representation.article_representation import ArticleRepresentation, parse_timestamp_to_epoch


def _make_rep(article_id: int, embedding: np.ndarray, published_at: str = "2026-11-23T12:00:00Z") -> ArticleRepresentation:
    epoch = parse_timestamp_to_epoch(published_at)
    return ArticleRepresentation(
        article_id=article_id,
        headline=f"Article {article_id}",
        description=f"Description {article_id}",
        source_id="test_source",
        source_name="Test Source",
        source_url=f"https://example.com/{article_id}",
        published_at=published_at,
        retrieved_at=published_at,
        published_epoch=epoch,
        text_for_embedding=f"Article {article_id}",
        text_hash=f"hash_{article_id}",
        embedding=embedding.astype(np.float32),
        entities={},
        normalized_entities={"entity_x"},
    )


class TestCentroidMathematics:
    """Rigorous mathematical tests for EventCluster running unnormalized sum and unit centroid."""

    def test_centroid_two_orthogonal_articles(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        cluster = EventCluster(
            id="evt_01",
            title="Test Event",
            summary="",
            first_seen_at="2026-11-23T10:00:00Z",
            last_seen_at="2026-11-23T10:00:00Z",
            article_ids=[1],
            article_count=1,
            centroid=v1.copy(),
            embedding_sum=v1.copy(),
        )

        cluster.add_article(_make_rep(2, v2, "2026-11-23T11:00:00Z"), {"token"})

        expected = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0], dtype=np.float32)
        np.testing.assert_allclose(cluster.centroid, expected, rtol=1e-5, atol=1e-5)
        assert pytest.approx(np.linalg.norm(cluster.centroid), abs=1e-6) == 1.0
        assert cluster.article_count == 2

    def test_centroid_three_or_more_articles(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        cluster = EventCluster(
            id="evt_02",
            title="Test Event",
            summary="",
            first_seen_at="2026-11-23T10:00:00Z",
            last_seen_at="2026-11-23T10:00:00Z",
            article_ids=[1],
            article_count=1,
            centroid=v1.copy(),
            embedding_sum=v1.copy(),
        )

        cluster.add_article(_make_rep(2, v2), {"t2"})
        cluster.add_article(_make_rep(3, v3), {"t3"})

        expected = np.array([1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)], dtype=np.float32)
        np.testing.assert_allclose(cluster.centroid, expected, rtol=1e-5, atol=1e-5)
        assert pytest.approx(np.linalg.norm(cluster.centroid), abs=1e-6) == 1.0
        assert cluster.article_count == 3

    def test_centroid_identical_vectors(self):
        v = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        v /= np.linalg.norm(v)

        cluster = EventCluster(
            id="evt_03",
            title="Identical Test",
            summary="",
            first_seen_at="2026-11-23T10:00:00Z",
            last_seen_at="2026-11-23T10:00:00Z",
            article_ids=[1],
            article_count=1,
            centroid=v.copy(),
            embedding_sum=v.copy(),
        )

        cluster.add_article(_make_rep(2, v), {"tok"})
        cluster.add_article(_make_rep(3, v), {"tok"})

        np.testing.assert_allclose(cluster.centroid, v, rtol=1e-5, atol=1e-5)
        assert pytest.approx(np.linalg.norm(cluster.centroid), abs=1e-6) == 1.0

    def test_centroid_opposing_vectors_degenerate_case(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)

        cluster = EventCluster(
            id="evt_04",
            title="Opposing Test",
            summary="",
            first_seen_at="2026-11-23T10:00:00Z",
            last_seen_at="2026-11-23T10:00:00Z",
            article_ids=[1],
            article_count=1,
            centroid=v1.copy(),
            embedding_sum=v1.copy(),
        )

        # Adding exact opposing vector produces S = [0, 0]
        cluster.add_article(_make_rep(2, v2), {"opp"})

        assert np.all(cluster.centroid == 0.0)
        np.testing.assert_allclose(cluster.embedding_sum, [0.0, 0.0], atol=1e-6)

    def test_incremental_updates_match_batch_arithmetic_mean(self):
        rng = np.random.default_rng(42)
        # Create 6 random unit vectors in 16 dimensions
        raw_vecs = [rng.normal(size=16).astype(np.float32) for _ in range(6)]
        vecs = [v / np.linalg.norm(v) for v in raw_vecs]

        # Batch unnormalized sum and normalized centroid
        batch_sum = np.sum(vecs, axis=0)
        expected_centroid = (batch_sum / np.linalg.norm(batch_sum)).astype(np.float32)

        # Incremental single-pass addition
        cluster = EventCluster(
            id="evt_batch_test",
            title="Batch Match Test",
            summary="",
            first_seen_at="2026-11-23T10:00:00Z",
            last_seen_at="2026-11-23T10:00:00Z",
            article_ids=[1],
            article_count=1,
            centroid=vecs[0].copy(),
            embedding_sum=vecs[0].copy(),
        )

        for idx, v in enumerate(vecs[1:], start=2):
            cluster.add_article(_make_rep(idx, v), {f"t_{idx}"})

        # Must match batch calculation to within machine float32 precision
        np.testing.assert_allclose(cluster.centroid, expected_centroid, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(cluster.embedding_sum, batch_sum, rtol=1e-5, atol=1e-5)

    def test_rebuild_centroid_method(self):
        v1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        cluster = EventCluster(
            id="evt_rebuild",
            title="Rebuild Test",
            summary="",
            first_seen_at="2026-11-23T10:00:00Z",
            last_seen_at="2026-11-23T10:00:00Z",
            article_ids=[1, 2],
            article_count=2,
            centroid=None,
            embedding_sum=None,
        )

        cluster.rebuild_centroid([v1, v2])
        assert cluster.centroid is not None
        assert pytest.approx(np.linalg.norm(cluster.centroid), abs=1e-6) == 1.0
        expected_sum = v1 + v2
        expected_c = expected_sum / np.linalg.norm(expected_sum)
        np.testing.assert_allclose(cluster.centroid, expected_c, rtol=1e-5)


class TestEventLifecycleSemantics:
    """Verifies complete deterministic partitioning of event lifecycle states across temporal intervals."""

    def test_single_article_is_emerging(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        t_pub = "2026-11-23T12:00:00Z"
        ref_epoch = parse_timestamp_to_epoch(t_pub) + (2.0 * 3600.0)  # 2 hours later

        cluster = EventCluster(
            id="evt_em",
            title="Emerging Test",
            summary="",
            first_seen_at=t_pub,
            last_seen_at=t_pub,
            article_ids=[1],
            article_count=1,
            centroid=v.copy(),
            embedding_sum=v.copy(),
            status="EMERGING",
        )

        status = cluster.update_lifecycle_status(reference_epoch=ref_epoch)
        assert status == "EMERGING"

    def test_two_articles_within_48h_is_active(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        t_pub1 = "2026-11-23T10:00:00Z"
        t_pub2 = "2026-11-23T14:00:00Z"
        ref_epoch = parse_timestamp_to_epoch(t_pub2) + (5.0 * 3600.0)  # 5h after second article

        cluster = EventCluster(
            id="evt_act",
            title="Active Test",
            summary="",
            first_seen_at=t_pub1,
            last_seen_at=t_pub1,
            article_ids=[1],
            article_count=1,
            centroid=v1.copy(),
            embedding_sum=v1.copy(),
            status="EMERGING",
        )

        cluster.add_article(_make_rep(2, v2, t_pub2), {"tok"}, reference_epoch=ref_epoch)
        assert cluster.status == "ACTIVE"

    def test_dormant_between_48h_and_168h(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        t_pub = "2026-11-20T12:00:00Z"
        # 60 hours later (between 48h and 168h)
        ref_epoch = parse_timestamp_to_epoch(t_pub) + (60.0 * 3600.0)

        cluster = EventCluster(
            id="evt_dorm",
            title="Dormant Test",
            summary="",
            first_seen_at=t_pub,
            last_seen_at=t_pub,
            article_ids=[1, 2],
            article_count=2,
            centroid=v.copy(),
            embedding_sum=v.copy() * 2.0,
            status="ACTIVE",
        )

        status = cluster.update_lifecycle_status(reference_epoch=ref_epoch)
        assert status == "DORMANT"

    def test_inactive_after_168h(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        t_pub = "2026-11-10T12:00:00Z"
        # 200 hours later (> 168h / 7 days)
        ref_epoch = parse_timestamp_to_epoch(t_pub) + (200.0 * 3600.0)

        cluster = EventCluster(
            id="evt_inact",
            title="Inactive Test",
            summary="",
            first_seen_at=t_pub,
            last_seen_at=t_pub,
            article_ids=[1, 2],
            article_count=2,
            centroid=v.copy(),
            embedding_sum=v.copy() * 2.0,
            status="ACTIVE",
        )

        status = cluster.update_lifecycle_status(reference_epoch=ref_epoch)
        assert status == "INACTIVE"

    def test_resolved_status_preserved(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        t_pub = "2026-11-23T12:00:00Z"
        cluster = EventCluster(
            id="evt_res",
            title="Resolved Test",
            summary="",
            first_seen_at=t_pub,
            last_seen_at=t_pub,
            article_ids=[1, 2],
            article_count=2,
            centroid=v.copy(),
            embedding_sum=v.copy() * 2.0,
            status="RESOLVED",
        )

        # Even with recent reference time, explicit RESOLVED status is never overwritten
        ref_epoch = parse_timestamp_to_epoch(t_pub) + 3600.0
        status = cluster.update_lifecycle_status(reference_epoch=ref_epoch)
        assert status == "RESOLVED"
