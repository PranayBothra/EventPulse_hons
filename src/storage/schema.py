"""SQLite DDL and schema management for EventPulse."""

CREATE_FEEDS_TABLE = """
CREATE TABLE IF NOT EXISTS feeds (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    category TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_fetched_at TEXT,
    last_status TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

CREATE_ARTICLES_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_url TEXT UNIQUE NOT NULL,
    url TEXT NOT NULL,
    headline TEXT NOT NULL,
    headline_clean TEXT NOT NULL,
    headline_hash TEXT NOT NULL,
    description TEXT,
    content_clean TEXT,
    source_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    is_sample INTEGER NOT NULL DEFAULT 0,
    raw_metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    dataset_origin TEXT NOT NULL DEFAULT 'event_corpus',
    content_availability TEXT NOT NULL DEFAULT 'HEADLINE_AND_DESCRIPTION',
    language TEXT DEFAULT 'en',
    country TEXT,
    seen_at TEXT,
    timestamp_precision TEXT NOT NULL DEFAULT 'UNKNOWN',
    timestamp_source TEXT,
    timestamp_timezone TEXT DEFAULT 'UTC',
    timestamp_is_inferred INTEGER NOT NULL DEFAULT 0,
    published_at_original TEXT,
    provider TEXT NOT NULL DEFAULT 'rss',
    external_id TEXT
);
"""

CREATE_BENCHMARK_NEWS_TABLE = """
CREATE TABLE IF NOT EXISTS benchmark_news (
    news_id TEXT PRIMARY KEY,
    category TEXT,
    subcategory TEXT,
    title TEXT NOT NULL,
    abstract TEXT,
    url TEXT,
    title_entities TEXT,
    abstract_entities TEXT,
    split TEXT NOT NULL DEFAULT 'train',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

CREATE_BENCHMARK_IMPRESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS benchmark_impressions (
    impression_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    time TEXT,
    history TEXT,
    impressions TEXT,
    split TEXT NOT NULL DEFAULT 'train',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (impression_id, split)
);
"""

CREATE_INGESTION_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    source TEXT NOT NULL,
    items_fetched INTEGER NOT NULL DEFAULT 0,
    items_stored INTEGER NOT NULL DEFAULT 0,
    items_duplicate INTEGER NOT NULL DEFAULT 0,
    items_malformed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_message TEXT
);
"""

CREATE_ARTICLE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS article_embeddings (
    article_id INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (article_id, model_id),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    article_count INTEGER NOT NULL DEFAULT 1,
    centroid_blob BLOB,
    key_entities TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

CREATE_EVENT_ARTICLES_TABLE = """
CREATE TABLE IF NOT EXISTS event_articles (
    event_id TEXT NOT NULL,
    article_id INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    is_ambiguous INTEGER NOT NULL DEFAULT 0,
    assigned_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (event_id, article_id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_ARTICLE_SENTENCES_TABLE = """
CREATE TABLE IF NOT EXISTS article_sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    sentence_index INTEGER NOT NULL,
    sentence_hash TEXT NOT NULL,
    sentence_text TEXT NOT NULL,
    is_information_bearing INTEGER NOT NULL DEFAULT 1,
    informativeness_score REAL NOT NULL DEFAULT 1.0,
    named_entities TEXT,
    has_numbers INTEGER NOT NULL DEFAULT 0,
    has_dates INTEGER NOT NULL DEFAULT 0,
    numerical_tokens TEXT,
    token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
    UNIQUE(article_id, sentence_index)
);
"""

CREATE_SENTENCE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS sentence_embeddings (
    sentence_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (sentence_id, model_id),
    FOREIGN KEY (sentence_id) REFERENCES article_sentences(id) ON DELETE CASCADE
);
"""

CREATE_EVENT_MEMORY_SENTENCES_TABLE = """
CREATE TABLE IF NOT EXISTS event_memory_sentences (
    event_id TEXT NOT NULL,
    sentence_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    published_at TEXT NOT NULL,
    published_epoch REAL NOT NULL,
    novelty_score REAL NOT NULL,
    evidence_status TEXT NOT NULL,
    most_similar_sentence_id INTEGER,
    highest_similarity REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (event_id, sentence_id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
    FOREIGN KEY (sentence_id) REFERENCES article_sentences(id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_ARTICLE_NOVELTY_TABLE = """
CREATE TABLE IF NOT EXISTS article_novelty (
    article_id INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    event_novelty_score REAL NOT NULL,
    novel_sentence_count INTEGER NOT NULL,
    redundant_sentence_count INTEGER NOT NULL,
    total_sentence_count INTEGER NOT NULL,
    novelty_ratio REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (article_id, event_id),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);
"""

CREATE_USER_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    long_term_profile_blob BLOB,
    short_term_profile_blob BLOB,
    explicit_topics TEXT,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    last_interaction_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

CREATE_USER_INTERACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS user_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    article_id INTEGER,
    event_id TEXT,
    interaction_type TEXT NOT NULL,
    interaction_weight REAL NOT NULL DEFAULT 1.0,
    interaction_time TEXT NOT NULL,
    interaction_epoch REAL NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
);
"""

CREATE_USER_CONSUMED_SENTENCES_TABLE = """
CREATE TABLE IF NOT EXISTS user_consumed_sentences (
    user_id TEXT NOT NULL,
    sentence_id INTEGER NOT NULL,
    consumed_at TEXT NOT NULL,
    consumed_epoch REAL NOT NULL,
    PRIMARY KEY (user_id, sentence_id),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
    FOREIGN KEY (sentence_id) REFERENCES article_sentences(id) ON DELETE CASCADE
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_canonical_url ON articles(canonical_url);",
    "CREATE INDEX IF NOT EXISTS idx_articles_headline_hash ON articles(headline_hash);",
    "CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles(source_id);",
    "CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_articles_is_sample ON articles(is_sample);",
    "CREATE INDEX IF NOT EXISTS idx_articles_dataset_origin ON articles(dataset_origin);",
    "CREATE INDEX IF NOT EXISTS idx_articles_provider_external_id ON articles(provider, external_id);",
    "CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at ON ingestion_runs(started_at);",
    "CREATE INDEX IF NOT EXISTS idx_article_embeddings_model ON article_embeddings(model_id);",
    "CREATE INDEX IF NOT EXISTS idx_events_last_seen_at ON events(last_seen_at);",
    "CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);",
    "CREATE INDEX IF NOT EXISTS idx_event_articles_article_id ON event_articles(article_id);",
    "CREATE INDEX IF NOT EXISTS idx_event_articles_event_id ON event_articles(event_id);",
    "CREATE INDEX IF NOT EXISTS idx_benchmark_news_category ON benchmark_news(category);",
    "CREATE INDEX IF NOT EXISTS idx_benchmark_news_split ON benchmark_news(split);",
    "CREATE INDEX IF NOT EXISTS idx_benchmark_impressions_user ON benchmark_impressions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_benchmark_impressions_split ON benchmark_impressions(split);",
    "CREATE INDEX IF NOT EXISTS idx_sentences_article ON article_sentences(article_id);",
    "CREATE INDEX IF NOT EXISTS idx_sentences_hash ON article_sentences(sentence_hash);",
    "CREATE INDEX IF NOT EXISTS idx_sentences_infobearing ON article_sentences(is_information_bearing);",
    "CREATE INDEX IF NOT EXISTS idx_sentence_embs_model ON sentence_embeddings(model_id);",
    "CREATE INDEX IF NOT EXISTS idx_event_mem_event ON event_memory_sentences(event_id);",
    "CREATE INDEX IF NOT EXISTS idx_event_mem_epoch ON event_memory_sentences(published_epoch);",
    "CREATE INDEX IF NOT EXISTS idx_art_nov_event ON article_novelty(event_id);",
    "CREATE INDEX IF NOT EXISTS idx_user_int_user ON user_interactions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_user_int_time ON user_interactions(interaction_epoch);",
    "CREATE INDEX IF NOT EXISTS idx_user_consumed_user ON user_consumed_sentences(user_id);",
]

ALL_TABLE_STATEMENTS = [
    CREATE_FEEDS_TABLE,
    CREATE_ARTICLES_TABLE,
    CREATE_INGESTION_RUNS_TABLE,
    CREATE_ARTICLE_EMBEDDINGS_TABLE,
    CREATE_EVENTS_TABLE,
    CREATE_EVENT_ARTICLES_TABLE,
    CREATE_BENCHMARK_NEWS_TABLE,
    CREATE_BENCHMARK_IMPRESSIONS_TABLE,
    CREATE_ARTICLE_SENTENCES_TABLE,
    CREATE_SENTENCE_EMBEDDINGS_TABLE,
    CREATE_EVENT_MEMORY_SENTENCES_TABLE,
    CREATE_ARTICLE_NOVELTY_TABLE,
    CREATE_USER_PROFILES_TABLE,
    CREATE_USER_INTERACTIONS_TABLE,
    CREATE_USER_CONSUMED_SENTENCES_TABLE,
]

ALL_SCHEMA_STATEMENTS = ALL_TABLE_STATEMENTS + CREATE_INDEXES
