"""Unit tests for EventPulse CLI interface."""

import pytest
from src.cli import main, parse_args
from src.storage.repository import SQLiteRepository


def test_parse_args_defaults():
    args = parse_args(["ingest"])
    assert args.command == "ingest"
    assert args.feed_id is None
    assert args.use_sample is False


def test_parse_args_options():
    args = parse_args(["ingest", "--feed", "bbc_world", "--sample", "--db", "custom.db"])
    assert args.command == "ingest"
    assert args.feed_id == "bbc_world"
    assert args.use_sample is True
    assert args.db_path == "custom.db"


def test_cli_ingest_sample_and_stats(temp_db: SQLiteRepository, capsys):
    db_path = str(temp_db.db_path)

    # Run ingest with sample
    exit_code = main(["ingest", "--sample", "--db", db_path])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "EVENTPULSE INGESTION SUMMARY" in captured.out
    assert "Overall Status   : SUCCESS" in captured.out

    # Run stats
    exit_code_stats = main(["stats", "--db", db_path])
    assert exit_code_stats == 0
    captured_stats = capsys.readouterr()
    assert "EVENTPULSE DATABASE STATISTICS" in captured_stats.out
    assert "Total Articles" in captured_stats.out

    # Run list-articles
    exit_code_list = main(["list-articles", "--db", db_path, "--limit", "5"])
    assert exit_code_list == 0
    captured_list = capsys.readouterr()
    assert "RECENT ARTICLES IN DATABASE" in captured_list.out
