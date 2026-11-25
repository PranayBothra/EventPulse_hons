"""Stage 3.9 integrity tests verifying leak-free external benchmark splits and label mappings."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_tap_dlnd import normalize_dla_label

WCEP_DIR = ROOT / "data" / "external_benchmarks" / "wcep"
TAP_DIR = ROOT / "data" / "external_benchmarks" / "tap_dlnd"


def test_tap_dlnd_label_normalization_rules():
    assert normalize_dla_label("Novel") == ("NOVEL", "POTENTIALLY_NEW", 1)
    assert normalize_dla_label("novel") == ("NOVEL", "POTENTIALLY_NEW", 1)
    assert normalize_dla_label("Non-Novel") == ("NON-NOVEL", "REPEATED_INFORMATION", 0)
    assert normalize_dla_label("non-novel") == ("NON-NOVEL", "REPEATED_INFORMATION", 0)
    assert normalize_dla_label("NovNon-Novelel") is None
    assert normalize_dla_label("Non-Novelvel") is None
    assert normalize_dla_label(None) is None


@pytest.mark.skipif(not (WCEP_DIR / "prepared_test_inputs.jsonl").exists(), reason="WCEP prepared files not present")
def test_wcep_prepared_splits_leak_free_and_disjoint():
    forbidden_keys = {"summary", "wiki_links", "reference_urls", "events", "ground_truth_cluster_id", "cluster_id", "date", "category"}
    val_urls = set()
    val_missing_time = 0
    val_count = 0
    with open(WCEP_DIR / "prepared_val_inputs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            val_count += 1
            assert not (forbidden_keys & set(rec.keys()))
            if rec.get("url"):
                val_urls.add(rec["url"])
            if rec.get("time") is None:
                val_missing_time += 1

    test_urls = set()
    test_missing_time = 0
    test_count = 0
    with open(WCEP_DIR / "prepared_test_inputs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            test_count += 1
            assert not (forbidden_keys & set(rec.keys()))
            if rec.get("url"):
                test_urls.add(rec["url"])
            if rec.get("time") is None:
                test_missing_time += 1

    assert val_count == 3160
    assert test_count == 6644
    assert len(val_urls & test_urls) == 0
    assert val_missing_time > 0
    assert test_missing_time > 0

    with open(WCEP_DIR / "prepared_val_labels.json", "r", encoding="utf-8") as f:
        val_clusters = {v["ground_truth_cluster_id"] for v in json.load(f).values()}
    with open(WCEP_DIR / "prepared_test_labels.json", "r", encoding="utf-8") as f:
        test_clusters = {v["ground_truth_cluster_id"] for v in json.load(f).values()}
    assert len(val_clusters & test_clusters) == 0


@pytest.mark.skipif(not (TAP_DIR / "prepared_test_inputs.jsonl").exists(), reason="TAP-DLND prepared files not present")
def test_tap_dlnd_event_disjoint_and_strict_labels():
    forbidden_input_keys = {"DLA", "raw_dla", "raw_label", "mapped_label", "binary_label"}
    split_events = {}
    excluded_ids = {"PLTE096TGT040", "TERE009TGT048", "TERE011TGT028"}

    for sp in ("train", "val", "test"):
        evs = set()
        in_ids = set()
        with open(TAP_DIR / f"prepared_{sp}_inputs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                assert not (forbidden_input_keys & set(rec.keys()))
                assert rec["doc_id"] not in excluded_ids
                evs.add(rec["event_id"])
                in_ids.add(rec["doc_id"])

        with open(TAP_DIR / f"prepared_{sp}_labels.json", "r", encoding="utf-8") as f:
            labels = json.load(f)

        assert set(labels.keys()) == in_ids
        for meta in labels.values():
            assert meta["mapped_label"] in {"POTENTIALLY_NEW", "REPEATED_INFORMATION"}
            assert meta["binary_label"] in {0, 1}
        split_events[sp] = evs

    assert len(split_events["train"] & split_events["val"]) == 0
    assert len(split_events["train"] & split_events["test"]) == 0
    assert len(split_events["val"] & split_events["test"]) == 0
