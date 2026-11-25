"""Command-line interface for EventPulse."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import load_config, load_feeds
from src.ingestion.pipeline import IngestionPipeline
from src.storage.repository import SQLiteRepository
from src.utils.logging import setup_logger


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="eventpulse",
        description="EventPulse: Temporal Novelty-Aware Event Intelligence and Personalised News Recommendation",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default="config/config.yaml",
        help="Path to global config.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest articles from RSS feeds")
    ingest_parser.add_argument(
        "--feed",
        dest="feed_id",
        default=None,
        help="Specific feed ID to ingest (default: all enabled feeds)",
    )
    ingest_parser.add_argument(
        "--sample",
        dest="use_sample",
        action="store_true",
        help="Ingest explicitly labelled sample fallback dataset instead of live feeds",
    )
    ingest_parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Override SQLite database path",
    )

    # Command: stats
    stats_parser = subparsers.add_parser("stats", help="Display database statistics")
    stats_parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Override SQLite database path",
    )

    # Command: check-feeds
    subparsers.add_parser(
        "check-feeds", help="Verify network reachability of configured RSS feeds"
    )

    # Command: list-articles
    list_parser = subparsers.add_parser(
        "list-articles", help="List recent ingested articles from database"
    )
    list_parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=10,
        help="Maximum articles to display",
    )
    list_parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Override SQLite database path",
    )

    # Command: cluster-events
    cluster_parser = subparsers.add_parser(
        "cluster-events", help="Perform temporal event clustering on database articles"
    )
    cluster_parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Override SQLite database path",
    )
    cluster_parser.add_argument(
        "--include-sample",
        dest="include_sample",
        action="store_true",
        help="Include synthetic sample fallback records (default: live articles only)",
    )
    cluster_parser.add_argument(
        "--provider",
        dest="provider",
        default=None,
        help="Override embedding provider ('sentence-transformers' or 'tfidf')",
    )

    # Command: list-events
    events_parser = subparsers.add_parser(
        "list-events", help="List formed real-world events and their member articles"
    )
    events_parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Override SQLite database path",
    )
    events_parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=20,
        help="Maximum events to display",
    )

    # Command: generate-validation-pairs
    pairs_parser = subparsers.add_parser(
        "generate-validation-pairs",
        help="Generate reviewable article pairs CSV for manual annotation",
    )
    pairs_parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Override SQLite database path",
    )
    pairs_parser.add_argument(
        "--output",
        dest="output_path",
        default="data/validation_pairs_blind.csv",
        help="Path for generated blinded validation CSV",
    )
    pairs_parser.add_argument(
        "--count",
        "--pairs",
        dest="count",
        type=int,
        default=35,
        help="Target number of validation pairs to sample",
    )

    # Command: sensitivity-analysis
    sens_parser = subparsers.add_parser(
        "sensitivity-analysis",
        help="Run threshold-sensitivity experiments on clustering parameters",
    )
    sens_parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Override SQLite database path",
    )

    return parser.parse_args(args)


def run_ingest(args: argparse.Namespace) -> int:
    """Execute the news ingestion command."""
    cfg = load_config(args.config_path)
    if args.db_path:
        cfg.database.path = args.db_path

    logger = setup_logger(
        level=cfg.logging.level,
        log_file=cfg.logging.log_file,
    )
    logger.info("Initializing EventPulse ingestion pipeline...")

    repo = SQLiteRepository(db_path=cfg.database.path)
    pipeline = IngestionPipeline(config=cfg, repository=repo)

    summary = pipeline.run(
        feed_id=args.feed_id,
        use_sample_fallback=args.use_sample,
    )

    print("\n" + "=" * 60)
    print("EVENTPULSE INGESTION SUMMARY")
    print("=" * 60)
    print(f"Overall Status   : {summary.overall_status}")
    print(f"Feeds Processed  : {summary.total_feeds_processed}")
    print(f"Articles Fetched : {summary.total_items_fetched}")
    print(f"Articles Stored  : {summary.total_items_stored}")
    print(f"Duplicates       : {summary.total_items_duplicate}")
    print(f"Malformed Items  : {summary.total_items_malformed}")
    print("-" * 60)
    for res in summary.feed_results:
        print(
            f"  [{res.status}] {res.feed_name} ({res.feed_id}): "
            f"{res.items_stored} stored, {res.items_duplicate} dup, {res.items_malformed} malformed"
        )
    print("=" * 60 + "\n")

    return 0 if summary.overall_status in ("SUCCESS", "PARTIAL") else 1


def run_stats(args: argparse.Namespace) -> int:
    """Display current database statistics."""
    cfg = load_config(args.config_path)
    db_path = args.db_path or cfg.database.path
    repo = SQLiteRepository(db_path=db_path)
    stats = repo.get_database_stats()

    print("\n" + "=" * 60)
    print("EVENTPULSE DATABASE STATISTICS")
    print("=" * 60)
    print(f"Database Path    : {stats['db_path']}")
    print(f"File Size        : {stats['db_size_bytes'] / 1024:.1f} KB")
    print(f"Total Articles   : {stats['total_articles']}")
    print(f"  - Live News    : {stats['live_articles']}")
    print(f"  - Sample Data  : {stats['sample_articles']}")
    print(f"Total Events     : {stats.get('total_events', 0)}")
    print(f"Cached Embeddings: {stats.get('total_cached_embeddings', 0)}")
    for m_id, m_cnt in stats.get("embeddings_by_model", {}).items():
        print(f"  - {m_id}: {m_cnt}")
    print(f"Feeds Registered : {stats['total_feeds']}")
    print(f"Ingestion Runs   : {stats['total_ingestion_runs']}")
    print("-" * 60)
    print("Articles by Source:")
    for src, cnt in stats["sources"].items():
        print(f"  - {src}: {cnt}")
    print("=" * 60 + "\n")
    return 0


def run_check_feeds(args: argparse.Namespace) -> int:
    """Verify connectivity to all configured feeds."""
    import urllib.request

    cfg = load_config(args.config_path)
    feeds = load_feeds(cfg.ingestion.feeds_file)
    print("\n" + "=" * 60)
    print("VERIFYING CONFIGURED RSS FEEDS")
    print("=" * 60)

    headers = {"User-Agent": cfg.ingestion.user_agent}
    all_ok = True

    for feed in feeds:
        try:
            req = urllib.request.Request(feed.url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                code = resp.status
                content_type = resp.headers.get("Content-Type", "unknown")
                status_str = "OK" if code == 200 else f"HTTP {code}"
                print(f"  [PASS] {feed.name} ({feed.id}) - {status_str} | {content_type}")
        except Exception as e:
            all_ok = False
            print(f"  [FAIL] {feed.name} ({feed.id}) - Error: {e}")

    print("=" * 60 + "\n")
    return 0 if all_ok else 1


def run_list_articles(args: argparse.Namespace) -> int:
    """List recent articles from the SQLite database."""
    cfg = load_config(args.config_path)
    db_path = args.db_path or cfg.database.path
    repo = SQLiteRepository(db_path=db_path)
    articles = repo.get_articles(limit=args.limit)

    print("\n" + "=" * 70)
    print(f"RECENT ARTICLES IN DATABASE (Showing {len(articles)})")
    print("=" * 70)
    for i, a in enumerate(articles, 1):
        sample_tag = "[SAMPLE] " if a.get("is_sample") else ""
        print(f"{i:2d}. {sample_tag}{a['headline']}")
        print(f"    Source: {a['source_name']} | Published: {a['published_at']}")
        print(f"    URL: {a['canonical_url']}")
        print()
    print("=" * 70 + "\n")
    return 0


def run_cluster_events(args: argparse.Namespace) -> int:
    """Execute temporal event clustering on database articles."""
    from src.clustering.engine import TemporalEventClusteringEngine
    from src.representation.embedding import get_embedding_provider

    cfg = load_config(args.config_path)
    db_path = args.db_path or cfg.database.path
    repo = SQLiteRepository(db_path=db_path)

    provider_name = args.provider or cfg.embedding.provider
    embed_provider = get_embedding_provider(
        provider_type=provider_name,
        model_name=cfg.embedding.model_name,
    )

    logger = setup_logger(level=cfg.logging.level, log_file=cfg.logging.log_file)
    logger.info("Initializing Temporal Event Clustering Engine (Provider: %s)...", provider_name)

    engine = TemporalEventClusteringEngine(
        config=cfg,
        repository=repo,
        embedding_provider=embed_provider,
    )

    is_sample_filter = None if args.include_sample else False
    decisions = engine.cluster_unassigned_articles(is_sample=is_sample_filter)

    new_events = sum(1 for d in decisions if d.is_new_event)
    assigned_existing = sum(1 for d in decisions if not d.is_new_event)
    ambiguous_count = sum(1 for d in decisions if d.is_ambiguous)

    all_events = repo.get_all_events(limit=500)
    multi_article_events = [e for e in all_events if e["article_count"] > 1]

    print("\n" + "=" * 65)
    print("EVENTPULSE TEMPORAL CLUSTERING SUMMARY")
    print("=" * 65)
    print(f"Articles Processed     : {len(decisions)}")
    print(f"New Events Formed      : {new_events}")
    print(f"Assigned to Existing   : {assigned_existing}")
    print(f"Ambiguous Assignments  : {ambiguous_count}")
    print(f"Total Persistent Events: {len(all_events)}")
    print(f"Multi-Article Clusters : {len(multi_article_events)}")
    print("-" * 65)
    print("Top Multi-Source Emerging Events:")
    for ev in multi_article_events[:8]:
        ents = ", ".join(ev["key_entities"][:4]) if ev["key_entities"] else "None"
        print(f"  * [{ev['article_count']} arts] {ev['title']}")
        print(f"    ID: {ev['id']} | Entities: [{ents}] | Status: {ev['status']}")
    print("=" * 65 + "\n")
    return 0


def run_list_events(args: argparse.Namespace) -> int:
    """List formed events and their assigned articles."""
    cfg = load_config(args.config_path)
    db_path = args.db_path or cfg.database.path
    repo = SQLiteRepository(db_path=db_path)
    events = repo.get_all_events(limit=args.limit)

    print("\n" + "=" * 75)
    print(f"EVENTPULSE PERSISTENT EVENTS (Showing {len(events)})")
    print("=" * 75)
    for i, ev in enumerate(events, 1):
        ents = ", ".join(ev["key_entities"][:4]) if ev["key_entities"] else "None"
        print(f"{i:2d}. [{ev['status']}] {ev['title']} ({ev['article_count']} articles)")
        print(f"    Event ID : {ev['id']}")
        print(f"    Timespan : {ev['first_seen_at']} -> {ev['last_seen_at']}")
        print(f"    Entities : [{ents}]")

        # Show member articles if multi-article event
        if ev["article_count"] > 1:
            members = repo.get_event_articles(ev["id"])
            print("    Member Articles:")
            for m in members:
                ambig = " [AMBIGUOUS]" if m.get("is_ambiguous") else ""
                print(f"      - ({m['source_id']}) {m['headline']} (Sim: {m['similarity_score']:.3f}){ambig}")
        print()
    print("=" * 75 + "\n")
    return 0


def run_generate_validation_pairs(args: argparse.Namespace) -> int:
    """Generate blinded manual validation CSV and separate predictions CSV."""
    from src.evaluation.pair_generator import generate_validation_pairs_csv
    from src.representation.embedding import get_embedding_provider

    cfg = load_config(args.config_path)
    db_path = args.db_path or cfg.database.path
    repo = SQLiteRepository(db_path=db_path)
    embedder = get_embedding_provider(
        provider_type=cfg.embedding.provider,
        model_name=cfg.embedding.model_name,
    )

    blind_path, pred_path = generate_validation_pairs_csv(
        repository=repo,
        embedding_provider=embedder,
        output_path=args.output_path,
        sample_pairs_count=args.count,
    )
    print(f"\n[SUCCESS] Generated blinded validation pairs CSV at: {blind_path}")
    print(f"[SUCCESS] Stored separate model predictions at   : {pred_path}\n")
    return 0


def run_sensitivity_analysis(args: argparse.Namespace) -> int:
    """Execute clustering sensitivity analysis over varying match thresholds."""
    from src.evaluation.sensitivity import run_threshold_sensitivity
    from src.representation.embedding import get_embedding_provider

    cfg = load_config(args.config_path)
    db_path = args.db_path or cfg.database.path
    repo = SQLiteRepository(db_path=db_path)
    embedder = get_embedding_provider(
        provider_type=cfg.embedding.provider,
        model_name=cfg.embedding.model_name,
    )

    thresholds = [0.35, 0.40, 0.45, 0.48, 0.52, 0.55, 0.60, 0.65]
    print("\n" + "=" * 80)
    print("RUNNING CLUSTERING THRESHOLD SENSITIVITY EXPERIMENT")
    print("=" * 80)
    results = run_threshold_sensitivity(
        repository=repo,
        embedding_provider=embedder,
        thresholds=thresholds,
        is_sample=False,
    )

    print(
        f"{'Threshold':<10} | {'Events':<7} | {'Multi':<6} | {'Single':<7} | {'Max Size':<9} | {'Avg Size':<9} | {'Ambiguous':<9}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['threshold']:<10.2f} | {r['total_events']:<7} | {r['multi_article_events']:<6} | {r['singletons']:<7} | {r['max_cluster_size']:<9} | {r['avg_cluster_size']:<9.2f} | {r['ambiguous_count']:<9}"
        )
    print("=" * 80 + "\n")
    return 0


def main(args: Optional[list] = None) -> int:
    """CLI main entry point."""
    parsed = parse_args(args)
    if not parsed.command:
        print("EventPulse CLI: No command specified. Use --help for usage.")
        return 1

    if parsed.command == "ingest":
        return run_ingest(parsed)
    elif parsed.command == "stats":
        return run_stats(parsed)
    elif parsed.command == "check-feeds":
        return run_check_feeds(parsed)
    elif parsed.command == "list-articles":
        return run_list_articles(parsed)
    elif parsed.command == "cluster-events":
        return run_cluster_events(parsed)
    elif parsed.command == "list-events":
        return run_list_events(parsed)
    elif parsed.command == "generate-validation-pairs":
        return run_generate_validation_pairs(parsed)
    elif parsed.command == "sensitivity-analysis":
        return run_sensitivity_analysis(parsed)
    else:
        print(f"Unknown command: {parsed.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
