"""SQLite storage repository for EventPulse."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.storage.schema import ALL_SCHEMA_STATEMENTS, ALL_TABLE_STATEMENTS, CREATE_INDEXES


class SQLiteRepository:
    """Thread-safe SQLite repository with idempotency and duplicate prevention."""

    def __init__(self, db_path: str = "data/eventpulse.db", timeout: int = 30):
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a new thread-local SQLite connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        # Enable foreign keys, busy timeout, and WAL performance pragmas
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)};")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def initialize_schema(self) -> None:
        """Execute all schema DDL and index creation statements with migration."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for stmt in ALL_TABLE_STATEMENTS:
                cursor.execute(stmt)

            # Additive migration for articles table columns
            cursor.execute("PRAGMA table_info(articles);")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            migrations = [
                ("dataset_origin", "TEXT NOT NULL DEFAULT 'event_corpus'"),
                ("content_availability", "TEXT NOT NULL DEFAULT 'HEADLINE_AND_DESCRIPTION'"),
                ("language", "TEXT DEFAULT 'en'"),
                ("country", "TEXT"),
                ("seen_at", "TEXT"),
                ("timestamp_precision", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
                ("timestamp_source", "TEXT"),
                ("timestamp_timezone", "TEXT DEFAULT 'UTC'"),
                ("timestamp_is_inferred", "INTEGER NOT NULL DEFAULT 0"),
                ("published_at_original", "TEXT"),
                ("provider", "TEXT NOT NULL DEFAULT 'rss'"),
                ("external_id", "TEXT"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_def};")

            for stmt in CREATE_INDEXES:
                cursor.execute(stmt)

            conn.commit()

    def upsert_feed(
        self,
        feed_id: str,
        name: str,
        url: str,
        category: str = "general",
        enabled: bool = True,
    ) -> None:
        """Register or update an RSS feed configuration."""
        query = """
        INSERT INTO feeds (id, name, url, category, enabled)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            url = excluded.url,
            category = excluded.category,
            enabled = excluded.enabled;
        """
        with self._get_connection() as conn:
            conn.execute(query, (feed_id, name, url, category, 1 if enabled else 0))
            conn.commit()

    def update_feed_status(
        self, feed_id: str, status: str, fetched_at: Optional[str] = None
    ) -> None:
        """Update last fetch status and timestamp for a feed."""
        timestamp = fetched_at or datetime.now(timezone.utc).isoformat()
        query = """
        UPDATE feeds
        SET last_status = ?, last_fetched_at = ?
        WHERE id = ?;
        """
        with self._get_connection() as conn:
            conn.execute(query, (status, timestamp, feed_id))
            conn.commit()

    def is_duplicate_url(self, canonical_url: str) -> bool:
        """Check whether an article with this canonical URL already exists."""
        query = "SELECT 1 FROM articles WHERE canonical_url = ? LIMIT 1;"
        with self._get_connection() as conn:
            cur = conn.execute(query, (canonical_url,))
            return cur.fetchone() is not None

    def is_duplicate_headline(self, headline_hash: str) -> bool:
        """Check whether an article with this headline hash already exists."""
        query = "SELECT 1 FROM articles WHERE headline_hash = ? LIMIT 1;"
        with self._get_connection() as conn:
            cur = conn.execute(query, (headline_hash,))
            return cur.fetchone() is not None

    def insert_article(self, article: Dict[str, Any]) -> bool:
        """Insert a single article into the database.

        Returns:
            True if inserted; False if duplicate (skipped).
        """
        query = """
        INSERT OR IGNORE INTO articles (
            canonical_url,
            url,
            headline,
            headline_clean,
            headline_hash,
            description,
            content_clean,
            source_id,
            source_name,
            published_at,
            retrieved_at,
            is_sample,
            raw_metadata,
            created_at,
            dataset_origin,
            content_availability,
            language,
            country,
            seen_at,
            timestamp_precision,
            timestamp_source,
            timestamp_timezone,
            timestamp_is_inferred,
            published_at_original,
            provider,
            external_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        now = datetime.now(timezone.utc).isoformat()
        raw_meta = article.get("raw_metadata")
        if isinstance(raw_meta, (dict, list)):
            raw_meta = json.dumps(raw_meta)

        params = (
            article["canonical_url"],
            article["url"],
            article["headline"],
            article["headline_clean"],
            article["headline_hash"],
            article.get("description", ""),
            article.get("content_clean", ""),
            article["source_id"],
            article.get("source_name", article["source_id"]),
            article.get("published_at"),
            article.get("retrieved_at", now),
            1 if article.get("is_sample", False) else 0,
            raw_meta,
            now,
            article.get("dataset_origin", "event_corpus"),
            article.get("content_availability", "HEADLINE_AND_DESCRIPTION"),
            article.get("language", "en"),
            article.get("country"),
            article.get("seen_at"),
            article.get("timestamp_precision", "UNKNOWN"),
            article.get("timestamp_source"),
            article.get("timestamp_timezone", "UTC"),
            1 if article.get("timestamp_is_inferred") else 0,
            article.get("published_at_original", article.get("published_at")),
            article.get("provider", "rss"),
            article.get("external_id"),
        )

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    def batch_insert_articles(
        self, articles: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """Insert multiple articles in a single atomic transaction.

        Returns:
            Tuple of (stored_count, duplicate_count).
        """
        stored = 0
        duplicates = 0
        now = datetime.now(timezone.utc).isoformat()

        query = """
        INSERT OR IGNORE INTO articles (
            canonical_url,
            url,
            headline,
            headline_clean,
            headline_hash,
            description,
            content_clean,
            source_id,
            source_name,
            published_at,
            retrieved_at,
            is_sample,
            raw_metadata,
            created_at,
            dataset_origin,
            content_availability,
            language,
            country,
            seen_at,
            timestamp_precision,
            timestamp_source,
            timestamp_timezone,
            timestamp_is_inferred,
            published_at_original,
            provider,
            external_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for article in articles:
                raw_meta = article.get("raw_metadata")
                if isinstance(raw_meta, (dict, list)):
                    raw_meta = json.dumps(raw_meta)

                params = (
                    article["canonical_url"],
                    article["url"],
                    article["headline"],
                    article["headline_clean"],
                    article["headline_hash"],
                    article.get("description", ""),
                    article.get("content_clean", ""),
                    article["source_id"],
                    article.get("source_name", article["source_id"]),
                    article.get("published_at"),
                    article.get("retrieved_at", now),
                    1 if article.get("is_sample", False) else 0,
                    raw_meta,
                    now,
                    article.get("dataset_origin", "event_corpus"),
                    article.get("content_availability", "HEADLINE_AND_DESCRIPTION"),
                    article.get("language", "en"),
                    article.get("country"),
                    article.get("seen_at"),
                    article.get("timestamp_precision", "UNKNOWN"),
                    article.get("timestamp_source"),
                    article.get("timestamp_timezone", "UTC"),
                    1 if article.get("timestamp_is_inferred") else 0,
                    article.get("published_at_original", article.get("published_at")),
                    article.get("provider", "rss"),
                    article.get("external_id"),
                )
                cursor.execute(query, params)
                if cursor.rowcount > 0:
                    stored += 1
                else:
                    duplicates += 1
            conn.commit()

        return stored, duplicates

    def update_article_timestamp_metadata(
        self,
        article_id: int,
        timestamp_precision: str,
        timestamp_source: Optional[str] = None,
        timestamp_timezone: str = "UTC",
        timestamp_is_inferred: bool = False,
        published_at_original: Optional[str] = None,
    ) -> None:
        """Update explicit timestamp metadata for a specific article."""
        query = """
        UPDATE articles SET
            timestamp_precision = ?,
            timestamp_source = ?,
            timestamp_timezone = ?,
            timestamp_is_inferred = ?,
            published_at_original = ?
        WHERE id = ?;
        """
        with self._get_connection() as conn:
            conn.execute(
                query,
                (
                    timestamp_precision,
                    timestamp_source,
                    timestamp_timezone,
                    1 if timestamp_is_inferred else 0,
                    published_at_original,
                    article_id,
                ),
            )
            conn.commit()

    def batch_update_timestamp_metadata(
        self, updates: List[Dict[str, Any]]
    ) -> int:
        """Batch update explicit timestamp metadata across multiple articles.
        
        updates items: {
            'article_id': int,
            'timestamp_precision': str,
            'timestamp_source': Optional[str],
            'timestamp_timezone': str,
            'timestamp_is_inferred': bool,
            'published_at_original': Optional[str]
        }
        """
        if not updates:
            return 0
        query = """
        UPDATE articles SET
            timestamp_precision = ?,
            timestamp_source = ?,
            timestamp_timezone = ?,
            timestamp_is_inferred = ?,
            published_at_original = ?
        WHERE id = ?;
        """
        params = [
            (
                u["timestamp_precision"],
                u.get("timestamp_source"),
                u.get("timestamp_timezone", "UTC"),
                1 if u.get("timestamp_is_inferred") else 0,
                u.get("published_at_original"),
                u["article_id"],
            )
            for u in updates
        ]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params)
            conn.commit()
            return cursor.rowcount

    def get_article_by_id(self, article_id: int) -> Optional[Dict[str, Any]]:
        """Fetch article by primary key ID."""
        query = "SELECT * FROM articles WHERE id = ?;"
        with self._get_connection() as conn:
            row = conn.execute(query, (article_id,)).fetchone()
            return dict(row) if row else None

    def get_article_by_canonical_url(
        self, canonical_url: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch article by canonical URL."""
        query = "SELECT * FROM articles WHERE canonical_url = ?;"
        with self._get_connection() as conn:
            row = conn.execute(query, (canonical_url,)).fetchone()
            return dict(row) if row else None

    def get_articles(
        self,
        limit: int = 100,
        offset: int = 0,
        source_id: Optional[str] = None,
        is_sample: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch a paginated list of articles."""
        conditions = []
        params: List[Any] = []

        if source_id is not None:
            conditions.append("source_id = ?")
            params.append(source_id)

        if is_sample is not None:
            conditions.append("is_sample = ?")
            params.append(1 if is_sample else 0)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
        SELECT * FROM articles
        {where_clause}
        ORDER BY published_at DESC, id DESC
        LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_article_count(
        self,
        source_id: Optional[str] = None,
        is_sample: Optional[bool] = None,
    ) -> int:
        """Count articles matching optional filters."""
        conditions = []
        params: List[Any] = []

        if source_id is not None:
            conditions.append("source_id = ?")
            params.append(source_id)

        if is_sample is not None:
            conditions.append("is_sample = ?")
            params.append(1 if is_sample else 0)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT COUNT(*) FROM articles {where_clause};"

        with self._get_connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def get_unassigned_article_count(self, is_sample: Optional[bool] = False) -> int:
        """Count articles not yet assigned to any event cluster."""
        conditions = ["ea.article_id IS NULL"]
        params: List[Any] = []
        if is_sample is not None:
            conditions.append("a.is_sample = ?")
            params.append(1 if is_sample else 0)
        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT COUNT(*)
        FROM articles a
        LEFT JOIN event_articles ea ON a.id = ea.article_id
        WHERE {where_clause};
        """
        with self._get_connection() as conn:
            return int(conn.execute(query, params).fetchone()[0])

    def record_ingestion_run(self, run_data: Dict[str, Any]) -> int:
        """Record an ingestion pipeline execution run."""
        query = """
        INSERT INTO ingestion_runs (
            started_at,
            completed_at,
            source,
            items_fetched,
            items_stored,
            items_duplicate,
            items_malformed,
            status,
            error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            run_data["started_at"],
            run_data.get("completed_at"),
            run_data["source"],
            run_data.get("items_fetched", 0),
            run_data.get("items_stored", 0),
            run_data.get("items_duplicate", 0),
            run_data.get("items_malformed", 0),
            run_data["status"],
            run_data.get("error_message"),
        )
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def get_latest_ingestion_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch the most recent ingestion runs."""
        query = """
        SELECT * FROM ingestion_runs
        ORDER BY id DESC
        LIMIT ?;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_database_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the database."""
        with self._get_connection() as conn:
            total_articles = conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0]
            live_articles = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE is_sample = 0;"
            ).fetchone()[0]
            sample_articles = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE is_sample = 1;"
            ).fetchone()[0]
            total_feeds = conn.execute("SELECT COUNT(*) FROM feeds;").fetchone()[0]
            total_runs = conn.execute(
                "SELECT COUNT(*) FROM ingestion_runs;"
            ).fetchone()[0]
            total_events = conn.execute("SELECT COUNT(*) FROM events;").fetchone()[0]
            total_cached_embeddings = conn.execute(
                "SELECT COUNT(*) FROM article_embeddings;"
            ).fetchone()[0]
            embeddings_by_model = conn.execute(
                "SELECT model_id, COUNT(*) as cnt FROM article_embeddings GROUP BY model_id ORDER BY cnt DESC;"
            ).fetchall()
            embeddings_breakdown = {row["model_id"]: row["cnt"] for row in embeddings_by_model}

            # Sources breakdown
            sources = conn.execute(
                "SELECT source_id, COUNT(*) as cnt FROM articles GROUP BY source_id ORDER BY cnt DESC;"
            ).fetchall()
            source_breakdown = {row["source_id"]: row["cnt"] for row in sources}

        db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size_bytes,
            "total_articles": total_articles,
            "live_articles": live_articles,
            "sample_articles": sample_articles,
            "total_feeds": total_feeds,
            "total_ingestion_runs": total_runs,
            "total_events": total_events,
            "total_cached_embeddings": total_cached_embeddings,
            "embeddings_by_model": embeddings_breakdown,
            "sources": source_breakdown,
        }

    # =========================================================================
    # Stage 2 Additive Methods: Embedding Cache & Event Storage
    # =========================================================================

    def save_embedding(
        self,
        article_id: int,
        text_hash: str,
        model_id: str,
        dimension: int,
        embedding_blob: bytes,
    ) -> None:
        """Store or update a cached dense vector representation."""
        query = """
        INSERT INTO article_embeddings (
            article_id, text_hash, model_id, dimension, embedding_blob, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id, model_id) DO UPDATE SET
            text_hash = excluded.text_hash,
            dimension = excluded.dimension,
            embedding_blob = excluded.embedding_blob,
            created_at = excluded.created_at;
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                query,
                (article_id, text_hash, model_id, dimension, embedding_blob, now),
            )
            conn.commit()

    def batch_save_embeddings(
        self,
        embeddings_data: List[Tuple[int, str, str, int, bytes]],
    ) -> int:
        """Store or update multiple cached embeddings in a single atomic transaction.

        Args:
            embeddings_data: List of (article_id, text_hash, model_id, dimension, embedding_blob).

        Returns:
            Number of embeddings saved.
        """
        if not embeddings_data:
            return 0
        query = """
        INSERT INTO article_embeddings (
            article_id, text_hash, model_id, dimension, embedding_blob, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id, model_id) DO UPDATE SET
            text_hash = excluded.text_hash,
            dimension = excluded.dimension,
            embedding_blob = excluded.embedding_blob,
            created_at = excluded.created_at;
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for art_id, th, m_id, dim, blob in embeddings_data:
                cursor.execute(query, (art_id, th, m_id, dim, blob, now))
            conn.commit()
        return len(embeddings_data)

    def get_embedding(
        self, article_id: int, text_hash: str, model_id: str
    ) -> Optional[bytes]:
        """Fetch cached embedding blob if text_hash and model_id match."""
        query = """
        SELECT embedding_blob FROM article_embeddings
        WHERE article_id = ? AND text_hash = ? AND model_id = ?;
        """
        with self._get_connection() as conn:
            row = conn.execute(query, (article_id, text_hash, model_id)).fetchone()
            return row["embedding_blob"] if row else None

    def get_embedding_by_article_id(
        self, article_id: int, model_id: Optional[str] = None
    ) -> Optional[bytes]:
        """Fetch cached embedding blob by article ID, optionally filtered by model."""
        if model_id:
            query = "SELECT embedding_blob FROM article_embeddings WHERE article_id = ? AND model_id = ?;"
            params = (article_id, model_id)
        else:
            query = "SELECT embedding_blob FROM article_embeddings WHERE article_id = ? ORDER BY created_at DESC LIMIT 1;"
            params = (article_id,)
        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return row["embedding_blob"] if row else None

    def get_embeddings_by_article_ids(
        self, article_ids: List[int], model_id: Optional[str] = None
    ) -> Dict[int, bytes]:
        """Fetch cached embedding blobs for a list of article IDs in a single query."""
        if not article_ids:
            return {}
        placeholders = ",".join("?" for _ in article_ids)
        if model_id:
            query = f"SELECT article_id, embedding_blob FROM article_embeddings WHERE article_id IN ({placeholders}) AND model_id = ?;"
            params = list(article_ids) + [model_id]
        else:
            query = f"SELECT article_id, embedding_blob FROM article_embeddings WHERE article_id IN ({placeholders});"
            params = list(article_ids)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return {row["article_id"]: row["embedding_blob"] for row in rows}

    def create_event(
        self,
        event_id: str,
        title: str,
        summary: Optional[str],
        first_seen_at: str,
        last_seen_at: str,
        centroid_blob: Optional[bytes],
        key_entities: Optional[List[str]],
        status: str = "ACTIVE",
    ) -> None:
        """Insert a newly detected real-world event."""
        query = """
        INSERT INTO events (
            id, title, summary, first_seen_at, last_seen_at,
            article_count, centroid_blob, key_entities, status,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?);
        """
        now = datetime.now(timezone.utc).isoformat()
        entities_json = json.dumps(key_entities or [])
        with self._get_connection() as conn:
            conn.execute(
                query,
                (
                    event_id,
                    title,
                    summary or "",
                    first_seen_at,
                    last_seen_at,
                    centroid_blob,
                    entities_json,
                    status,
                    now,
                    now,
                ),
            )
            conn.commit()

    def update_event(
        self,
        event_id: str,
        last_seen_at: str,
        article_count: int,
        centroid_blob: Optional[bytes],
        key_entities: Optional[List[str]],
        status: Optional[str] = None,
    ) -> None:
        """Update an existing event with new centroid, entities, and latest timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        entities_json = json.dumps(key_entities or [])

        set_clauses = [
            "last_seen_at = ?",
            "article_count = ?",
            "centroid_blob = ?",
            "key_entities = ?",
            "updated_at = ?",
        ]
        params: List[Any] = [
            last_seen_at,
            article_count,
            centroid_blob,
            entities_json,
            now,
        ]

        if status is not None:
            set_clauses.append("status = ?")
            params.append(status)

        params.append(event_id)
        query = f"UPDATE events SET {', '.join(set_clauses)} WHERE id = ?;"

        with self._get_connection() as conn:
            conn.execute(query, params)
            conn.commit()

    def assign_article_to_event(
        self,
        event_id: str,
        article_id: int,
        similarity_score: float,
        is_ambiguous: bool = False,
    ) -> None:
        """Assign an article to an event record."""
        query = """
        INSERT OR REPLACE INTO event_articles (
            event_id, article_id, similarity_score, is_ambiguous, assigned_at
        ) VALUES (?, ?, ?, ?, ?);
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                query,
                (
                    event_id,
                    article_id,
                    float(similarity_score),
                    1 if is_ambiguous else 0,
                    now,
                ),
            )
            conn.commit()

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Fetch event by primary key identifier."""
        query = "SELECT * FROM events WHERE id = ?;"
        with self._get_connection() as conn:
            row = conn.execute(query, (event_id,)).fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("key_entities"):
                try:
                    res["key_entities"] = json.loads(res["key_entities"])
                except Exception:
                    res["key_entities"] = []
            return res

    def get_all_events(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve events with optional status filtering."""
        params: List[Any] = []
        where_clause = ""
        if status:
            where_clause = "WHERE status = ?"
            params.append(status)

        query = f"""
        SELECT * FROM events
        {where_clause}
        ORDER BY last_seen_at DESC
        LIMIT ?;
        """
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            events = []
            for r in rows:
                item = dict(r)
                if item.get("key_entities"):
                    try:
                        item["key_entities"] = json.loads(item["key_entities"])
                    except Exception:
                        item["key_entities"] = []
                events.append(item)
            return events

    def get_event_articles(self, event_id: str) -> List[Dict[str, Any]]:
        """Fetch all articles assigned to a given event, ordered chronologically."""
        query = """
        SELECT a.*, ea.similarity_score, ea.is_ambiguous, ea.assigned_at
        FROM articles a
        JOIN event_articles ea ON a.id = ea.article_id
        WHERE ea.event_id = ?
        ORDER BY a.published_at ASC, a.id ASC;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, (event_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_event_for_article(self, article_id: int) -> Optional[Dict[str, Any]]:
        """Find the event assigned to an article."""
        query = """
        SELECT e.*, ea.similarity_score, ea.is_ambiguous
        FROM events e
        JOIN event_articles ea ON e.id = ea.event_id
        WHERE ea.article_id = ?;
        """
        with self._get_connection() as conn:
            row = conn.execute(query, (article_id,)).fetchone()
            if not row:
                return None
            item = dict(row)
            if item.get("key_entities"):
                try:
                    item["key_entities"] = json.loads(item["key_entities"])
                except Exception:
                    item["key_entities"] = []
            return item

    def batch_insert_benchmark_news(
        self, news_items: List[Dict[str, Any]]
    ) -> int:
        """Insert news items into benchmark_news table.

        Returns:
            Count of inserted items.
        """
        query = """
        INSERT OR IGNORE INTO benchmark_news (
            news_id,
            category,
            subcategory,
            title,
            abstract,
            url,
            title_entities,
            abstract_entities,
            split,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for item in news_items:
                cursor.execute(
                    query,
                    (
                        item["news_id"],
                        item.get("category"),
                        item.get("subcategory"),
                        item["title"],
                        item.get("abstract"),
                        item.get("url"),
                        item.get("title_entities"),
                        item.get("abstract_entities"),
                        item.get("split", "train"),
                        now,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
            conn.commit()
        return inserted

    def batch_insert_benchmark_impressions(
        self, impressions: List[Dict[str, Any]]
    ) -> int:
        """Insert impression interactions into benchmark_impressions table.

        Returns:
            Count of inserted rows.
        """
        query = """
        INSERT OR IGNORE INTO benchmark_impressions (
            impression_id,
            user_id,
            time,
            history,
            impressions,
            split,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for imp in impressions:
                cursor.execute(
                    query,
                    (
                        imp["impression_id"],
                        imp["user_id"],
                        imp.get("time"),
                        imp.get("history"),
                        imp.get("impressions"),
                        imp.get("split", "train"),
                        now,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
            conn.commit()
        return inserted

    def get_benchmark_stats(self) -> Dict[str, Any]:
        """Return counts and splits for benchmark datasets."""
        with self._get_connection() as conn:
            news_total = conn.execute("SELECT COUNT(*) FROM benchmark_news;").fetchone()[0]
            news_by_split = dict(
                conn.execute(
                    "SELECT split, COUNT(*) FROM benchmark_news GROUP BY split;"
                ).fetchall()
            )
            news_by_cat = dict(
                conn.execute(
                    "SELECT category, COUNT(*) FROM benchmark_news GROUP BY category ORDER BY COUNT(*) DESC LIMIT 10;"
                ).fetchall()
            )
            imp_total = conn.execute("SELECT COUNT(*) FROM benchmark_impressions;").fetchone()[0]
            imp_by_split = dict(
                conn.execute(
                    "SELECT split, COUNT(*) FROM benchmark_impressions GROUP BY split;"
                ).fetchall()
            )
            user_count = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM benchmark_impressions;"
            ).fetchone()[0]

            return {
                "news_total": news_total,
                "news_by_split": news_by_split,
                "top_categories": news_by_cat,
                "impressions_total": imp_total,
                "impressions_by_split": imp_by_split,
                "unique_users": user_count,
            }

    # =========================================================================
    # STAGE 3: NOVELTY & USER MODELLING METHODS
    # =========================================================================

    def save_article_sentences(self, sentences: List[Dict[str, Any]]) -> List[int]:
        """Save segmented and filtered sentences for an article.
        
        Returns the list of generated sentence IDs.
        """
        if not sentences:
            return []
        query = """
        INSERT INTO article_sentences (
            article_id, sentence_index, sentence_hash, sentence_text,
            is_information_bearing, informativeness_score, named_entities,
            has_numbers, has_dates, numerical_tokens, token_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id, sentence_index) DO UPDATE SET
            sentence_hash=excluded.sentence_hash,
            sentence_text=excluded.sentence_text,
            is_information_bearing=excluded.is_information_bearing,
            informativeness_score=excluded.informativeness_score,
            named_entities=excluded.named_entities,
            has_numbers=excluded.has_numbers,
            has_dates=excluded.has_dates,
            numerical_tokens=excluded.numerical_tokens,
            token_count=excluded.token_count
        RETURNING id;
        """
        ids = []
        with self._get_connection() as conn:
            for s in sentences:
                cur = conn.execute(
                    query,
                    (
                        s["article_id"],
                        s["sentence_index"],
                        s["sentence_hash"],
                        s["sentence_text"],
                        1 if s.get("is_information_bearing", True) else 0,
                        float(s.get("informativeness_score", 1.0)),
                        json.dumps(list(s.get("named_entities", []))),
                        1 if s.get("has_numbers", False) else 0,
                        1 if s.get("has_dates", False) else 0,
                        json.dumps(list(s.get("numerical_tokens", []))),
                        int(s.get("token_count", 0)),
                    ),
                )
                row = cur.fetchone()
                if row:
                    ids.append(row[0])
            conn.commit()
        return ids

    def get_article_sentences(
        self, article_id: int, only_information_bearing: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve sentences belonging to an article."""
        query = """
        SELECT id, article_id, sentence_index, sentence_hash, sentence_text,
               is_information_bearing, informativeness_score, named_entities,
               has_numbers, has_dates, numerical_tokens, token_count, created_at
        FROM article_sentences
        WHERE article_id = ?
        """
        if only_information_bearing:
            query += " AND is_information_bearing = 1"
        query += " ORDER BY sentence_index ASC;"

        with self._get_connection() as conn:
            rows = conn.execute(query, (article_id,)).fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "article_id": r["article_id"],
                    "sentence_index": r["sentence_index"],
                    "sentence_hash": r["sentence_hash"],
                    "sentence_text": r["sentence_text"],
                    "is_information_bearing": bool(r["is_information_bearing"]),
                    "informativeness_score": float(r["informativeness_score"]),
                    "named_entities": set(json.loads(r["named_entities"])) if r["named_entities"] else set(),
                    "has_numbers": bool(r["has_numbers"]),
                    "has_dates": bool(r["has_dates"]),
                    "numerical_tokens": json.loads(r["numerical_tokens"]) if r["numerical_tokens"] else [],
                    "token_count": r["token_count"],
                    "created_at": r["created_at"],
                })
            return results

    def batch_save_sentence_embeddings(self, records: List[Tuple[int, str, bytes]]) -> int:
        """Batch insert or update sentence embeddings (sentence_id, model_id, blob)."""
        if not records:
            return 0
        query = """
        INSERT INTO sentence_embeddings (sentence_id, model_id, embedding_blob)
        VALUES (?, ?, ?)
        ON CONFLICT(sentence_id, model_id) DO UPDATE SET
            embedding_blob = excluded.embedding_blob,
            created_at = (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
        """
        with self._get_connection() as conn:
            conn.executemany(query, records)
            conn.commit()
        return len(records)

    def get_sentence_embeddings(
        self, sentence_ids: List[int], model_id: str
    ) -> Dict[int, bytes]:
        """Fetch sentence embeddings by IDs for a specific model."""
        if not sentence_ids:
            return {}
        placeholders = ",".join("?" for _ in sentence_ids)
        query = f"""
        SELECT sentence_id, embedding_blob
        FROM sentence_embeddings
        WHERE model_id = ? AND sentence_id IN ({placeholders});
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, [model_id] + list(sentence_ids)).fetchall()
            return {r["sentence_id"]: bytes(r["embedding_blob"]) for r in rows}

    def save_event_memory_sentences(self, memory_records: List[Dict[str, Any]]) -> int:
        """Persist information sentences associated with an event's historical memory."""
        if not memory_records:
            return 0
        query = """
        INSERT INTO event_memory_sentences (
            event_id, sentence_id, article_id, published_at, published_epoch,
            novelty_score, evidence_status, most_similar_sentence_id, highest_similarity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id, sentence_id) DO UPDATE SET
            novelty_score = excluded.novelty_score,
            evidence_status = excluded.evidence_status,
            most_similar_sentence_id = excluded.most_similar_sentence_id,
            highest_similarity = excluded.highest_similarity;
        """
        with self._get_connection() as conn:
            for r in memory_records:
                conn.execute(
                    query,
                    (
                        r["event_id"],
                        r["sentence_id"],
                        r["article_id"],
                        r["published_at"],
                        float(r["published_epoch"]),
                        float(r["novelty_score"]),
                        r["evidence_status"],
                        r.get("most_similar_sentence_id"),
                        float(r["highest_similarity"]) if r.get("highest_similarity") is not None else None,
                    ),
                )
            conn.commit()
        return len(memory_records)

    def get_event_memory_sentences(
        self, event_id: str, before_epoch: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve chronological information sentences for an event up to a specific epoch."""
        query = """
        SELECT em.event_id, em.sentence_id, em.article_id, em.published_at, em.published_epoch,
               em.novelty_score, em.evidence_status, em.most_similar_sentence_id, em.highest_similarity,
               s.sentence_text, s.named_entities, s.informativeness_score, s.has_numbers, s.has_dates,
               COALESCE(a.timestamp_precision, 'UNKNOWN') AS timestamp_precision,
               a.timestamp_source,
               COALESCE(a.timestamp_is_inferred, 0) AS timestamp_is_inferred,
               a.published_at_original
        FROM event_memory_sentences em
        JOIN article_sentences s ON em.sentence_id = s.id
        LEFT JOIN articles a ON em.article_id = a.id
        WHERE em.event_id = ?
        """
        params = [event_id]
        if before_epoch is not None:
            query += " AND em.published_epoch < ?"
            params.append(before_epoch)
        query += " ORDER BY em.published_epoch ASC, em.sentence_id ASC;"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                results.append({
                    "event_id": r["event_id"],
                    "sentence_id": r["sentence_id"],
                    "article_id": r["article_id"],
                    "published_at": r["published_at"],
                    "published_epoch": float(r["published_epoch"]),
                    "novelty_score": float(r["novelty_score"]),
                    "evidence_status": r["evidence_status"],
                    "most_similar_sentence_id": r["most_similar_sentence_id"],
                    "highest_similarity": float(r["highest_similarity"]) if r["highest_similarity"] is not None else None,
                    "sentence_text": r["sentence_text"],
                    "named_entities": set(json.loads(r["named_entities"])) if r["named_entities"] else set(),
                    "informativeness_score": float(r["informativeness_score"]),
                    "has_numbers": bool(r["has_numbers"]),
                    "has_dates": bool(r["has_dates"]),
                    "timestamp_precision": r["timestamp_precision"],
                    "timestamp_source": r["timestamp_source"],
                    "timestamp_is_inferred": bool(r["timestamp_is_inferred"]),
                    "published_at_original": r["published_at_original"],
                })
            return results

    def save_article_novelty(self, novelty_data: Dict[str, Any]) -> None:
        """Save aggregated article-level novelty scores."""
        query = """
        INSERT INTO article_novelty (
            article_id, event_id, event_novelty_score,
            novel_sentence_count, redundant_sentence_count, total_sentence_count, novelty_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id, event_id) DO UPDATE SET
            event_novelty_score = excluded.event_novelty_score,
            novel_sentence_count = excluded.novel_sentence_count,
            redundant_sentence_count = excluded.redundant_sentence_count,
            total_sentence_count = excluded.total_sentence_count,
            novelty_ratio = excluded.novelty_ratio;
        """
        with self._get_connection() as conn:
            conn.execute(
                query,
                (
                    novelty_data["article_id"],
                    novelty_data["event_id"],
                    float(novelty_data["event_novelty_score"]),
                    int(novelty_data["novel_sentence_count"]),
                    int(novelty_data["redundant_sentence_count"]),
                    int(novelty_data["total_sentence_count"]),
                    float(novelty_data["novelty_ratio"]),
                ),
            )
            conn.commit()

    def get_article_novelty(self, article_id: int, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve precomputed novelty score for an article."""
        query = """
        SELECT article_id, event_id, event_novelty_score,
               novel_sentence_count, redundant_sentence_count, total_sentence_count, novelty_ratio
        FROM article_novelty
        WHERE article_id = ? AND event_id = ?;
        """
        with self._get_connection() as conn:
            row = conn.execute(query, (article_id, event_id)).fetchone()
            if not row:
                return None
            return {
                "article_id": row["article_id"],
                "event_id": row["event_id"],
                "event_novelty_score": float(row["event_novelty_score"]),
                "novel_sentence_count": row["novel_sentence_count"],
                "redundant_sentence_count": row["redundant_sentence_count"],
                "total_sentence_count": row["total_sentence_count"],
                "novelty_ratio": float(row["novelty_ratio"]),
            }

    def upsert_user_profile(
        self,
        user_id: str,
        long_term_blob: Optional[bytes] = None,
        short_term_blob: Optional[bytes] = None,
        explicit_topics: Optional[List[str]] = None,
    ) -> None:
        """Create or update a user's persistent interest profiles."""
        query = """
        INSERT INTO user_profiles (
            user_id, long_term_profile_blob, short_term_profile_blob, explicit_topics, updated_at
        ) VALUES (?, ?, ?, ?, (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')))
        ON CONFLICT(user_id) DO UPDATE SET
            long_term_profile_blob = COALESCE(excluded.long_term_profile_blob, user_profiles.long_term_profile_blob),
            short_term_profile_blob = COALESCE(excluded.short_term_profile_blob, user_profiles.short_term_profile_blob),
            explicit_topics = COALESCE(excluded.explicit_topics, user_profiles.explicit_topics),
            updated_at = (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
        """
        topics_json = json.dumps(explicit_topics) if explicit_topics is not None else None
        with self._get_connection() as conn:
            conn.execute(query, (user_id, long_term_blob, short_term_blob, topics_json))
            conn.commit()

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile representation."""
        query = """
        SELECT user_id, long_term_profile_blob, short_term_profile_blob,
               explicit_topics, interaction_count, last_interaction_at, created_at, updated_at
        FROM user_profiles
        WHERE user_id = ?;
        """
        with self._get_connection() as conn:
            row = conn.execute(query, (user_id,)).fetchone()
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "long_term_profile_blob": bytes(row["long_term_profile_blob"]) if row["long_term_profile_blob"] else None,
                "short_term_profile_blob": bytes(row["short_term_profile_blob"]) if row["short_term_profile_blob"] else None,
                "explicit_topics": json.loads(row["explicit_topics"]) if row["explicit_topics"] else [],
                "interaction_count": row["interaction_count"],
                "last_interaction_at": row["last_interaction_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def record_user_interaction(
        self,
        user_id: str,
        article_id: Optional[int],
        event_id: Optional[str],
        interaction_type: str,
        weight: float,
        interaction_time: str,
        interaction_epoch: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Log a user interaction and increment interaction count."""
        query_interaction = """
        INSERT INTO user_interactions (
            user_id, article_id, event_id, interaction_type, interaction_weight,
            interaction_time, interaction_epoch, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        query_update_user = """
        INSERT INTO user_profiles (user_id, interaction_count, last_interaction_at, updated_at)
        VALUES (?, 1, ?, (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')))
        ON CONFLICT(user_id) DO UPDATE SET
            interaction_count = user_profiles.interaction_count + 1,
            last_interaction_at = excluded.last_interaction_at,
            updated_at = (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
        """
        meta_json = json.dumps(metadata) if metadata else None
        with self._get_connection() as conn:
            # Upsert user_profiles first to satisfy foreign key constraint on user_interactions
            conn.execute(query_update_user, (user_id, interaction_time))
            cur = conn.execute(
                query_interaction,
                (
                    user_id, article_id, event_id, interaction_type,
                    float(weight), interaction_time, float(interaction_epoch), meta_json
                ),
            )
            interaction_id = cur.lastrowid
            conn.commit()
            return interaction_id

    def get_user_interactions(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent interaction history for a user."""
        query = """
        SELECT id, user_id, article_id, event_id, interaction_type, interaction_weight,
               interaction_time, interaction_epoch, metadata, created_at
        FROM user_interactions
        WHERE user_id = ?
        ORDER BY interaction_epoch DESC, id DESC
        LIMIT ?;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, (user_id, limit)).fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "article_id": r["article_id"],
                    "event_id": r["event_id"],
                    "interaction_type": r["interaction_type"],
                    "interaction_weight": float(r["interaction_weight"]),
                    "interaction_time": r["interaction_time"],
                    "interaction_epoch": float(r["interaction_epoch"]),
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    "created_at": r["created_at"],
                })
            return results

    def record_user_consumed_sentences(
        self,
        user_id: str,
        sentence_ids: List[int],
        consumed_time: str,
        consumed_epoch: float,
    ) -> int:
        """Record that a user has consumed a set of sentences."""
        if not sentence_ids:
            return 0
        query = """
        INSERT OR IGNORE INTO user_consumed_sentences (
            user_id, sentence_id, consumed_at, consumed_epoch
        ) VALUES (?, ?, ?, ?);
        """
        with self._get_connection() as conn:
            for sid in sentence_ids:
                conn.execute(query, (user_id, sid, consumed_time, float(consumed_epoch)))
            conn.commit()
        return len(sentence_ids)

    def get_user_consumed_sentence_ids(self, user_id: str) -> Set[int]:
        """Fetch all sentence IDs consumed by a user."""
        query = "SELECT sentence_id FROM user_consumed_sentences WHERE user_id = ?;"
        with self._get_connection() as conn:
            rows = conn.execute(query, (user_id,)).fetchall()
            return {r[0] for r in rows}

    # =========================================================================
    # Stage 2 API Migration Methods (Extracted from UI & External Provider Support)
    # =========================================================================

    def get_existing_external_ids(
        self, provider: str, external_ids: List[str]
    ) -> Set[str]:
        """Return subset of external_ids already persisted for a given provider."""
        clean_ids = [eid for eid in external_ids if eid]
        if not clean_ids:
            return set()
        placeholders = ",".join("?" for _ in clean_ids)
        query = (
            f"SELECT external_id FROM articles "
            f"WHERE provider = ? AND external_id IN ({placeholders});"
        )
        with self._get_connection() as conn:
            rows = conn.execute(query, [provider] + clean_ids).fetchall()
            return {r["external_id"] for r in rows if r["external_id"]}

    def get_top_events_with_publisher_count(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        sort_by: str = "article_count",
    ) -> List[Dict[str, Any]]:
        """Retrieve events with distinct publisher count, matching Streamlit Event Explorer semantics."""
        params: List[Any] = []
        where_clause = ""
        if status:
            where_clause = "WHERE e.status = ?"
            params.append(status)

        order_expr = "e.last_seen_at DESC" if sort_by == "last_seen_at" else "e.article_count DESC, e.last_seen_at DESC"
        query = f"""
        SELECT e.id, e.title, e.summary, e.article_count, e.first_seen_at, e.last_seen_at,
               e.status, COUNT(DISTINCT a.source_id) AS publisher_count
        FROM events e
        JOIN event_articles ea ON ea.event_id = e.id
        JOIN articles a ON ea.article_id = a.id
        {where_clause}
        GROUP BY e.id
        ORDER BY {order_expr}
        LIMIT ?;
        """
        params.append(limit)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_event_articles_with_novelty(self, event_id: str) -> List[Dict[str, Any]]:
        """Retrieve chronological articles in an event joined with computed article novelty scores."""
        query = """
        SELECT a.id, a.headline, a.source_id, a.source_name, a.published_at,
               a.description, a.url, a.provider, a.external_id,
               an.event_novelty_score, an.novelty_ratio,
               ea.similarity_score, ea.is_ambiguous
        FROM event_articles ea
        JOIN articles a ON ea.article_id = a.id
        LEFT JOIN article_novelty an ON an.article_id = a.id AND an.event_id = ea.event_id
        WHERE ea.event_id = ?
        ORDER BY a.published_at ASC, a.id ASC;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, (event_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_evolving_events(
        self, min_articles: int = 2, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Retrieve multi-article evolving events eligible for novelty inspection."""
        query = """
        SELECT e.id, e.title, e.article_count, e.first_seen_at, e.last_seen_at, e.status
        FROM events e
        WHERE e.article_count >= ?
        ORDER BY e.article_count DESC, e.last_seen_at DESC
        LIMIT ?;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, (min_articles, limit)).fetchall()
            return [dict(r) for r in rows]
