"""Prepare leak-free, event-disjoint TAP-DLND 1.0 splits and generate dataset audit."""

import hashlib
import json
import logging
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_tap_dlnd")

ROOT = Path(__file__).resolve().parent.parent
TAP_BASE = ROOT / "data" / "external_benchmarks" / "tap_dlnd" / "extracted" / "TAP-DLND-1.0_LREC2018_modified"
OUT_DIR = ROOT / "data" / "external_benchmarks" / "tap_dlnd"
AUDIT_PATH = ROOT / "reports" / "tap_dlnd_dataset_audit.md"

SPLIT_SEED = 42


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _parse_xml_features(xml_path: Path) -> Dict[str, str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    attrs: Dict[str, str] = {}
    for k, v in root.attrib.items():
        attrs[f"_root_{k}"] = v.strip()
    for feat in root.findall("feature"):
        for k, v in feat.attrib.items():
            attrs[k] = v.strip()
    return attrs


def normalize_dla_label(raw_dla: Optional[str]) -> Optional[Tuple[str, str, int]]:
    """Map canonical TAP-DLND DLA attribute to EventPulse ground-truth binary target."""
    if not raw_dla:
        return None
    norm = raw_dla.strip().lower()
    if norm == "novel":
        return ("NOVEL", "POTENTIALLY_NEW", 1)
    if norm == "non-novel":
        return ("NON-NOVEL", "REPEATED_INFORMATION", 0)
    return None


def main() -> None:
    if not TAP_BASE.exists():
        raise FileNotFoundError(f"Extracted TAP-DLND directory not found at {TAP_BASE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    categories = sorted([d.name for d in TAP_BASE.iterdir() if d.is_dir()])
    logger.info(f"Scanning {len(categories)} TAP-DLND categories: {categories}")

    events: Dict[str, Dict[str, Any]] = {}
    raw_dla_counter: Counter = Counter()
    dop_counter: Counter = Counter()
    excluded_records: List[Dict[str, Any]] = []
    confusion_records: List[Dict[str, Any]] = []
    source_count = 0
    total_target_xml_count = 0
    valid_targets: List[Dict[str, Any]] = []

    duplicate_source_target_hashes = 0
    within_event_duplicate_target_hashes = 0

    for cat in categories:
        cat_dir = TAP_BASE / cat
        event_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
        for ev_dir in event_dirs:
            event_id = ev_dir.name
            src_dir = ev_dir / "source"
            tgt_dir = ev_dir / "target"

            sources: List[Dict[str, Any]] = []
            src_hashes = set()
            event_name = ""

            if src_dir.is_dir():
                for txt_path in sorted(src_dir.glob("*.txt"), key=lambda p: p.name):
                    doc_id = txt_path.stem
                    xml_path = src_dir / f"{doc_id}.xml"
                    text = _read_text(txt_path)
                    attrs = _parse_xml_features(xml_path) if xml_path.exists() else {}
                    if not event_name and attrs.get("eventname"):
                        event_name = attrs["eventname"]
                    dop = attrs.get("DOP", "")
                    dop_counter["placeholder_00" if (not dop or dop == "00/00/0000") else "dated"] += 1
                    sources.append({
                        "doc_id": doc_id,
                        "title": attrs.get("title", ""),
                        "text": text,
                        "dop": dop,
                        "word_count": len(text.split()),
                    })
                    src_hashes.add(hashlib.sha1(text.encode("utf-8")).hexdigest())
                    source_count += 1

            ev_targets: List[Dict[str, Any]] = []
            ev_tgt_hashes = set()

            if tgt_dir.is_dir():
                for entry in sorted(tgt_dir.iterdir(), key=lambda p: p.name):
                    if entry.is_dir() and entry.name.lower() == "confusion":
                        for cf in sorted(entry.iterdir(), key=lambda p: p.name):
                            confusion_records.append({
                                "category": cat,
                                "event_id": event_id,
                                "file": cf.name,
                                "reason": "Located in target/confusion ambiguity exclusion directory",
                            })
                        continue
                    if entry.is_file() and entry.suffix.lower() == ".xml":
                        total_target_xml_count += 1
                        doc_id = entry.stem
                        txt_path = tgt_dir / f"{doc_id}.txt"
                        attrs = _parse_xml_features(entry)
                        if not event_name and attrs.get("eventname"):
                            event_name = attrs["eventname"]
                        raw_dla = attrs.get("DLA")
                        raw_dla_counter[raw_dla if raw_dla is not None else "<MISSING>"] += 1

                        dop = attrs.get("DOP", "")
                        dop_counter["placeholder_00" if (not dop or dop == "00/00/0000") else "dated"] += 1

                        if not txt_path.exists():
                            excluded_records.append({
                                "category": cat,
                                "event_id": event_id,
                                "doc_id": doc_id,
                                "raw_dla": raw_dla,
                                "reason": "Missing matching .txt file",
                            })
                            continue

                        text = _read_text(txt_path)
                        if len(text) < 10:
                            excluded_records.append({
                                "category": cat,
                                "event_id": event_id,
                                "doc_id": doc_id,
                                "raw_dla": raw_dla,
                                "reason": f"Empty or sub-10-char target text (len={len(text)})",
                            })
                            continue

                        mapped = normalize_dla_label(raw_dla)
                        if mapped is None:
                            excluded_records.append({
                                "category": cat,
                                "event_id": event_id,
                                "doc_id": doc_id,
                                "raw_dla": raw_dla,
                                "reason": f"Malformed or missing DLA label: {raw_dla!r}",
                            })
                            continue

                        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                        if h in src_hashes:
                            duplicate_source_target_hashes += 1
                        if h in ev_tgt_hashes:
                            within_event_duplicate_target_hashes += 1
                        ev_tgt_hashes.add(h)

                        raw_norm, mapped_label, binary_label = mapped
                        rec = {
                            "doc_id": doc_id,
                            "event_id": event_id,
                            "category": cat,
                            "event_name": event_name,
                            "text": text,
                            "word_count": len(text.split()),
                            "raw_dla": raw_dla,
                            "raw_label": raw_norm,
                            "mapped_label": mapped_label,
                            "binary_label": binary_label,
                            "source_ids": [s["doc_id"] for s in sources],
                        }
                        ev_targets.append(rec)
                        valid_targets.append(rec)

            events[event_id] = {
                "event_id": event_id,
                "category": cat,
                "event_name": event_name,
                "sources": sources,
                "num_valid_targets": len(ev_targets),
                "num_novel": sum(1 for t in ev_targets if t["binary_label"] == 1),
                "num_non_novel": sum(1 for t in ev_targets if t["binary_label"] == 0),
            }

    # Event-disjoint stratified split: 50% train / 20% val / 30% test within each category
    rng = random.Random(SPLIT_SEED)
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for ev_id, info in sorted(events.items()):
        by_cat[info["category"]].append(ev_id)

    event_to_split: Dict[str, str] = {}
    for cat, ev_ids in sorted(by_cat.items()):
        shuffled = list(ev_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, round(0.20 * n))
        n_test = max(2, round(0.30 * n)) if n >= 4 else max(1, n - n_val - 1)
        if n_val + n_test >= n:
            n_train = max(1, n - 2)
            n_val = 1
            n_test = n - n_train - n_val
        else:
            n_train = n - n_val - n_test

        for ev in shuffled[:n_train]:
            event_to_split[ev] = "train"
        for ev in shuffled[n_train : n_train + n_val]:
            event_to_split[ev] = "val"
        for ev in shuffled[n_train + n_val :]:
            event_to_split[ev] = "test"

    # Assert strict event disjointness
    split_events: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for ev_id, sp in event_to_split.items():
        events[ev_id]["split"] = sp
        split_events[sp].add(ev_id)
    assert len(split_events["train"] & split_events["val"]) == 0
    assert len(split_events["train"] & split_events["test"]) == 0
    assert len(split_events["val"] & split_events["test"]) == 0

    with open(OUT_DIR / "prepared_events.json", "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    split_stats: Dict[str, Dict[str, Any]] = {}
    for sp in ("train", "val", "test"):
        sp_targets = [t for t in valid_targets if event_to_split[t["event_id"]] == sp]
        inputs_path = OUT_DIR / f"prepared_{sp}_inputs.jsonl"
        labels_path = OUT_DIR / f"prepared_{sp}_labels.json"
        labels_map: Dict[str, Any] = {}
        with open(inputs_path, "w", encoding="utf-8") as fin:
            for t in sp_targets:
                clean_in = {
                    "doc_id": t["doc_id"],
                    "event_id": t["event_id"],
                    "category": t["category"],
                    "text": t["text"],
                    "word_count": t["word_count"],
                    "source_ids": t["source_ids"],
                }
                fin.write(json.dumps(clean_in, ensure_ascii=False) + "\n")
                labels_map[t["doc_id"]] = {
                    "event_id": t["event_id"],
                    "category": t["category"],
                    "raw_dla": t["raw_dla"],
                    "raw_label": t["raw_label"],
                    "mapped_label": t["mapped_label"],
                    "binary_label": t["binary_label"],
                }
        with open(labels_path, "w", encoding="utf-8") as flab:
            json.dump(labels_map, flab, indent=2)

        n_nov = sum(1 for t in sp_targets if t["binary_label"] == 1)
        n_non = len(sp_targets) - n_nov
        split_stats[sp] = {
            "events": len(split_events[sp]),
            "targets": len(sp_targets),
            "novel": n_nov,
            "non_novel": n_non,
            "novel_rate": round(n_nov / len(sp_targets), 4) if sp_targets else 0.0,
        }

    # Compute per-category statistics
    cat_summary: Dict[str, Dict[str, Any]] = {}
    for cat in categories:
        cat_evs = [ev for ev, info in events.items() if info["category"] == cat]
        cat_tgts = [t for t in valid_targets if t["category"] == cat]
        n_nov = sum(1 for t in cat_tgts if t["binary_label"] == 1)
        cat_summary[cat] = {
            "events_total": len(cat_evs),
            "events_train": sum(1 for ev in cat_evs if event_to_split[ev] == "train"),
            "events_val": sum(1 for ev in cat_evs if event_to_split[ev] == "val"),
            "events_test": sum(1 for ev in cat_evs if event_to_split[ev] == "test"),
            "valid_targets": len(cat_tgts),
            "novel": n_nov,
            "non_novel": len(cat_tgts) - n_nov,
            "novel_rate": round(n_nov / len(cat_tgts), 4) if cat_tgts else 0.0,
        }

    # Write dataset audit markdown
    total_valid = len(valid_targets)
    total_novel = sum(1 for t in valid_targets if t["binary_label"] == 1)
    total_non_novel = total_valid - total_novel
    src_words = [s["word_count"] for info in events.values() for s in info["sources"]]
    tgt_words = [t["word_count"] for t in valid_targets]

    lines = [
        "# TAP-DLND 1.0 Dataset Audit & Event-Disjoint Split Report",
        "",
        "## 1. Benchmark Origin & Schema",
        "",
        "- **Dataset:** TAP-DLND 1.0 (Ghosal et al., LREC 2018; extracted from `TAP-DLND-1.0_LREC2018_04032018.zip` / `TAP-DLND-1.0_LREC2018_modified`).",
        "- **Task:** Document-level novelty classification against fixed event source background documents (`3` source articles per event).",
        f"- **Categories Discovered:** `{len(categories)}` (`{', '.join(categories)}`)",
        f"- **Total Events:** `{len(events)}`",
        f"- **Total Source Documents:** `{source_count}` (`{source_count / len(events):.2f}` per event)",
        f"- **Total Target XML Files (excluding `confusion/`):** `{total_target_xml_count}`",
        f"- **Valid Paired Targets Retained:** `{total_valid}`",
        f"- **Confusion Subdirectory Files Excluded:** `{len(confusion_records)}` (`{[(c['category'], c['event_id'], c['file']) for c in confusion_records]}`)",
        f"- **Malformed / Unpaired Targets Excluded:** `{len(excluded_records)}`",
        "",
        "## 2. Raw DLA Tag Audit & Exclusions",
        "",
        "Raw `DLA` attribute frequencies across all parsed target XML files:",
        "",
        "| Raw `DLA` Value | Count | Normalization / Action |",
        "| --- | ---: | --- |",
    ]
    for k, v in sorted(raw_dla_counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        norm = normalize_dla_label(None if k == "<MISSING>" else k)
        action = f"`{norm[0]}` -> `{norm[1]}`" if norm else "EXCLUDED (malformed/ambiguous)"
        lines.append(f"| `{k}` | {v} | {action} |")

    lines.extend([
        "",
        "Excluded malformed target files:",
    ])
    for ex in excluded_records:
        lines.append(f"- `{ex['category']}/{ex['event_id']}/{ex['doc_id']}`: {ex['reason']}")

    lines.extend([
        "",
        "## 3. EventPulse Label Mapping & Integrity Rules",
        "",
        "- `Novel` / `novel` -> **`NOVEL`** -> EventPulse `POTENTIALLY_NEW` (`binary_label = 1`, positive class: `n = " + str(total_novel) + f"`, `{100.0 * total_novel / total_valid:.2f}%`).",
        "- `Non-Novel` / `non-novel` -> **`NON-NOVEL`** -> EventPulse `REPEATED_INFORMATION` (`binary_label = 0`, negative class: `n = " + str(total_non_novel) + f"`, `{100.0 * total_non_novel / total_valid:.2f}%`).",
        "- **Strict Mapping Guardrail:** `BACKGROUND_INFORMATION` and `INSUFFICIENT_EVIDENCE` are **not** ground-truth labels in TAP-DLND and are never fabricated as ground truth. For binary classification evaluation, continuous novelty scores and discrete decisions are evaluated strictly against `POTENTIALLY_NEW` (1) vs `REPEATED_INFORMATION` (0).",
        f"- **Metadata & Leakage Checks:** DOP timestamp placeholders (`00/00/0000` or empty): `{dop_counter['placeholder_00']}`; dated entries: `{dop_counter['dated']}`. Exact source-target SHA-1 text matches: `{duplicate_source_target_hashes}`; within-event duplicate target texts: `{within_event_duplicate_target_hashes}`.",
        f"- **Length Statistics (words):** Sources mean `{sum(src_words)/len(src_words):.1f}` (min `{min(src_words)}`, max `{max(src_words)}`); Targets mean `{sum(tgt_words)/len(tgt_words):.1f}` (min `{min(tgt_words)}`, max `{max(tgt_words)}`).",
        "",
        "## 4. Event-Disjoint Stratified Splits (`seed=42`)",
        "",
        "All source documents and target articles belonging to an event are assigned strictly to a single split (`0` event overlap between train, validation, and test).",
        "",
        "| Split | Events | Targets | `POTENTIALLY_NEW` (`NOVEL`) | `REPEATED_INFORMATION` (`NON-NOVEL`) | Novel Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for sp in ("train", "val", "test"):
        st = split_stats[sp]
        lines.append(
            f"| `{sp}` | {st['events']} | {st['targets']} | {st['novel']} | {st['non_novel']} | {100.0 * st['novel_rate']:.2f}% |"
        )

    lines.extend([
        "",
        "### Per-Category Breakdown",
        "",
        "| Category | Events (Tr/Va/Te) | Valid Targets | `NOVEL` | `NON-NOVEL` | Novel Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for cat, cs in cat_summary.items():
        ev_str = f"{cs['events_total']} ({cs['events_train']}/{cs['events_val']}/{cs['events_test']})"
        lines.append(
            f"| `{cat}` | {ev_str} | {cs['valid_targets']} | {cs['novel']} | {cs['non_novel']} | {100.0 * cs['novel_rate']:.2f}% |"
        )

    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote TAP-DLND dataset audit to {AUDIT_PATH}")
    logger.info(f"Split summary: {split_stats}")


if __name__ == "__main__":
    main()
