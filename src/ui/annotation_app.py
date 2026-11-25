"""
EventPulse Double-Blind Human Annotation Application (Stage 3.5).
Lightweight Streamlit interface for independent multi-annotator evaluation of:
1. Event Clustering: Pairwise identity between news articles (SAME_EVENT, RELATED_EVENT, DIFFERENT_EVENT, INSUFFICIENT_EVIDENCE).
2. Sentence-Level Information Novelty: Proposition novelty against event memory (POTENTIALLY_NEW, REPEATED_INFORMATION, BACKGROUND_INFORMATION, INSUFFICIENT_EVIDENCE).

Strict Blinding Enforcement:
- ZERO model novelty scores, embedding similarities, cluster IDs, or heuristic triggers are exposed.
- Annotations are stored separately from model predictions in data/annotations/.
- Multiple independent annotators can submit without seeing each other's labels.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import streamlit as st

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLUSTERING_CSV = BASE_DIR / "data" / "event_clustering_validation_blind.csv"
NOVELTY_CSV = BASE_DIR / "data" / "novelty_validation_annotation_template.csv"
ANNOTATION_DIR = BASE_DIR / "data" / "annotations"
CLUSTERING_OUT = ANNOTATION_DIR / "event_clustering_annotations.csv"
NOVELTY_OUT = ANNOTATION_DIR / "novelty_annotations.csv"

CLUSTERING_LABELS = [
    "SAME_EVENT",
    "RELATED_EVENT",
    "DIFFERENT_EVENT",
    "INSUFFICIENT_EVIDENCE",
]

NOVELTY_LABELS = [
    "POTENTIALLY_NEW",
    "REPEATED_INFORMATION",
    "BACKGROUND_INFORMATION",
    "INSUFFICIENT_EVIDENCE",
]

CLUSTERING_DESCRIPTIONS = {
    "SAME_EVENT": "Articles describe the exact same real-world incident or occurrence.",
    "RELATED_EVENT": "Connected topic, ongoing broader story, or sub-event, but distinct occurrence.",
    "DIFFERENT_EVENT": "Separate, independent event with no direct causal or narrative identity.",
    "INSUFFICIENT_EVIDENCE": "Text is too vague, fragmented, or incomplete to judge.",
}

NOVELTY_DESCRIPTIONS = {
    "POTENTIALLY_NEW": "Candidate sentence provides genuinely new facts, updates, or outcome changes.",
    "REPEATED_INFORMATION": "Candidate sentence restates or paraphrases information already present in memory.",
    "BACKGROUND_INFORMATION": "General historical background, boilerplate, or descriptive context.",
    "INSUFFICIENT_EVIDENCE": "Sentence is too ambiguous or context is insufficient to determine novelty.",
}


def ensure_storage_ready():
    """Ensure data/annotations/ directory and CSV headers exist."""
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    if not CLUSTERING_OUT.exists():
        with open(CLUSTERING_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "blind_pair_id",
                "annotator_id",
                "human_label",
                "explanation",
                "annotation_timestamp",
            ])

    if not NOVELTY_OUT.exists():
        with open(NOVELTY_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "blind_pair_id",
                "annotator_id",
                "human_label",
                "explanation",
                "annotation_timestamp",
            ])


def load_existing_annotations(csv_path: Path, annotator_id: str) -> Dict[str, Dict[str, str]]:
    """Load existing submitted annotations for a specific annotator."""
    if not csv_path.exists():
        return {}
    res = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("annotator_id") == annotator_id:
                res[row["blind_pair_id"]] = row
    return res


def append_annotation(csv_path: Path, record: Dict[str, str]):
    """Safely append an annotation record to disk."""
    ensure_storage_ready()
    # Read all existing records to support non-destructive upsert per annotator
    all_rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or [
                "blind_pair_id", "annotator_id", "human_label", "explanation", "annotation_timestamp"
            ]
            for row in reader:
                if not (row["blind_pair_id"] == record["blind_pair_id"] and row["annotator_id"] == record["annotator_id"]):
                    all_rows.append(row)
    else:
        fieldnames = ["blind_pair_id", "annotator_id", "human_label", "explanation", "annotation_timestamp"]

    all_rows.append(record)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def main():
    st.set_page_config(
        page_title="EventPulse — Human Annotation Suite",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ensure_storage_ready()

    st.title("📰 EventPulse — Blind Human Ground-Truth Annotation")
    st.caption("Stage 3.5 Multi-Annotator Independent Research Validation Interface")

    # Sidebar: Session & Authentication
    st.sidebar.header("👤 Annotator Identity")
    annotator_id = st.sidebar.text_input(
        "Enter Your Annotator ID",
        value=st.session_state.get("annotator_id", ""),
        help="Unique identifier for independent validation (e.g. annotator_1, annotator_2).",
    ).strip()

    if annotator_id:
        st.session_state["annotator_id"] = annotator_id
    else:
        st.warning("⚠️ Please enter an Annotator ID in the sidebar to begin or resume annotation.")
        st.stop()

    task = st.sidebar.radio(
        "Select Annotation Task",
        ["Event Clustering Validation", "Sentence Novelty Validation"],
    )

    if task == "Event Clustering Validation":
        render_event_clustering(annotator_id)
    else:
        render_sentence_novelty(annotator_id)


def render_event_clustering(annotator_id: str):
    st.subheader("1. Event Clustering Pairwise Validation")
    st.info(
        "**Objective:** Determine whether Article A and Article B describe the exact same real-world event. "
        "All model similarity scores and cluster IDs are completely blinded."
    )

    if not CLUSTERING_CSV.exists():
        st.error(f"Missing validation file: {CLUSTERING_CSV}")
        return

    # Load pairs
    with open(CLUSTERING_CSV, "r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    total_pairs = len(pairs)
    existing_annotations = load_existing_annotations(CLUSTERING_OUT, annotator_id)
    annotated_count = len(existing_annotations)

    # Progress bar
    progress = annotated_count / max(1, total_pairs)
    st.sidebar.markdown(f"**Progress:** {annotated_count} / {total_pairs} ({progress * 100:.1f}%)")
    st.sidebar.progress(progress)

    # Navigation
    unannotated_indices = [i for i, p in enumerate(pairs) if p["blind_pair_id"] not in existing_annotations]
    default_idx = unannotated_indices[0] if unannotated_indices else 0

    if "ec_pair_idx" not in st.session_state:
        st.session_state["ec_pair_idx"] = default_idx

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅️ Previous Pair") and st.session_state["ec_pair_idx"] > 0:
            st.session_state["ec_pair_idx"] -= 1
            st.rerun()
    with col_nav2:
        pair_options = [f"Pair {i+1} ({p['blind_pair_id']})" + (" [DONE]" if p["blind_pair_id"] in existing_annotations else "") for i, p in enumerate(pairs)]
        selected_pair_str = st.selectbox(
            "Select Pair to Review",
            options=pair_options,
            index=st.session_state["ec_pair_idx"],
        )
        st.session_state["ec_pair_idx"] = pair_options.index(selected_pair_str)
    with col_nav3:
        if st.button("Next Pair ➡️") and st.session_state["ec_pair_idx"] < total_pairs - 1:
            st.session_state["ec_pair_idx"] += 1
            st.rerun()

    current_idx = st.session_state["ec_pair_idx"]
    pair = pairs[current_idx]
    pid = pair["blind_pair_id"]

    st.markdown(f"### Blind Item ID: `{pid}`")

    # Side-by-side article display
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📄 Article A")
        st.markdown(f"**Headline:**  \n### {pair['article_a_headline']}")
        st.markdown(f"**Publisher:** `{pair.get('article_a_publisher', 'Unknown')}`")
        st.markdown(f"**Published Date:** `{pair.get('article_a_published_at', 'Unknown')}`")

    with col_b:
        st.markdown("#### 📄 Article B")
        st.markdown(f"**Headline:**  \n### {pair['article_b_headline']}")
        st.markdown(f"**Publisher:** `{pair.get('article_b_publisher', 'Unknown')}`")
        st.markdown(f"**Published Date:** `{pair.get('article_b_published_at', 'Unknown')}`")

    if pair.get("annotation_notes"):
        st.caption(f"**Evaluation Context:** {pair['annotation_notes']}")

    st.divider()

    # Annotation form
    st.markdown("#### Select Ground-Truth Identity Label:")
    existing_entry = existing_annotations.get(pid, {})
    saved_label = existing_entry.get("human_label", None)
    saved_expl = existing_entry.get("explanation", "")

    default_radio_idx = CLUSTERING_LABELS.index(saved_label) if saved_label in CLUSTERING_LABELS else 0

    selected_label = st.radio(
        "Choose label:",
        CLUSTERING_LABELS,
        index=default_radio_idx,
        format_func=lambda lbl: f"{lbl} — {CLUSTERING_DESCRIPTIONS[lbl]}",
    )

    explanation = st.text_area(
        "Optional Reasoning / Notes (e.g. key shared entity, distinct date or outcome):",
        value=saved_expl,
    )

    if st.button("💾 Submit Annotation & Next", type="primary"):
        record = {
            "blind_pair_id": pid,
            "annotator_id": annotator_id,
            "human_label": selected_label,
            "explanation": explanation.strip(),
            "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_annotation(CLUSTERING_OUT, record)
        st.success(f"Annotation saved for {pid}!")

        # Advance to next unannotated if available
        next_unannotated = [i for i in range(current_idx + 1, total_pairs) if pairs[i]["blind_pair_id"] not in load_existing_annotations(CLUSTERING_OUT, annotator_id)]
        if next_unannotated:
            st.session_state["ec_pair_idx"] = next_unannotated[0]
        elif current_idx < total_pairs - 1:
            st.session_state["ec_pair_idx"] += 1
        st.rerun()


def render_sentence_novelty(annotator_id: str):
    st.subheader("2. Sentence-Level Information Novelty Validation")
    st.info(
        "**Objective:** Determine whether the candidate sentence conveys genuinely new information relative "
        "to the historical event memory context. All model novelty scores and thresholds are completely blinded."
    )

    if not NOVELTY_CSV.exists():
        st.error(f"Missing validation file: {NOVELTY_CSV}")
        return

    # Load pairs
    with open(NOVELTY_CSV, "r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    total_pairs = len(pairs)
    existing_annotations = load_existing_annotations(NOVELTY_OUT, annotator_id)
    annotated_count = len(existing_annotations)

    # Progress bar
    progress = annotated_count / max(1, total_pairs)
    st.sidebar.markdown(f"**Progress:** {annotated_count} / {total_pairs} ({progress * 100:.1f}%)")
    st.sidebar.progress(progress)

    # Navigation
    unannotated_indices = [i for i, p in enumerate(pairs) if p["blind_pair_id"] not in existing_annotations]
    default_idx = unannotated_indices[0] if unannotated_indices else 0

    if "nov_pair_idx" not in st.session_state:
        st.session_state["nov_pair_idx"] = default_idx

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅️ Previous Proposition") and st.session_state["nov_pair_idx"] > 0:
            st.session_state["nov_pair_idx"] -= 1
            st.rerun()
    with col_nav2:
        pair_options = [f"Item {i+1} ({p['blind_pair_id']})" + (" [DONE]" if p["blind_pair_id"] in existing_annotations else "") for i, p in enumerate(pairs)]
        selected_pair_str = st.selectbox(
            "Select Proposition Item",
            options=pair_options,
            index=st.session_state["nov_pair_idx"],
        )
        st.session_state["nov_pair_idx"] = pair_options.index(selected_pair_str)
    with col_nav3:
        if st.button("Next Proposition ➡️") and st.session_state["nov_pair_idx"] < total_pairs - 1:
            st.session_state["nov_pair_idx"] += 1
            st.rerun()

    current_idx = st.session_state["nov_pair_idx"]
    pair = pairs[current_idx]
    pid = pair["blind_pair_id"]

    st.markdown(f"### Blind Item ID: `{pid}`")

    # Historical Context vs Candidate Sentence
    st.markdown("#### 📜 Historical Event Proposition (Previously Known in Event Memory)")
    st.info(f"\"{pair.get('sentence_historical', '')}\"")

    st.markdown("#### 🆕 Candidate Proposition (Incoming New Information)")
    st.warning(f"\"{pair.get('sentence_candidate', '')}\"")

    if pair.get("event_context_id"):
        st.caption(f"**Event Group Identifier:** `{pair['event_context_id']}`")

    st.divider()

    # Annotation form
    st.markdown("#### Select Ground-Truth Information Novelty Label:")
    existing_entry = existing_annotations.get(pid, {})
    saved_label = existing_entry.get("human_label", None)
    saved_expl = existing_entry.get("explanation", "")

    default_radio_idx = NOVELTY_LABELS.index(saved_label) if saved_label in NOVELTY_LABELS else 0

    selected_label = st.radio(
        "Choose novelty category:",
        NOVELTY_LABELS,
        index=default_radio_idx,
        format_func=lambda lbl: f"{lbl} — {NOVELTY_DESCRIPTIONS[lbl]}",
    )

    explanation = st.text_area(
        "Optional Reasoning (e.g. numerical update, polarity flip, identical paraphrase, or boilerplate credit):",
        value=saved_expl,
    )

    if st.button("💾 Submit Novelty Annotation & Next", type="primary"):
        record = {
            "blind_pair_id": pid,
            "annotator_id": annotator_id,
            "human_label": selected_label,
            "explanation": explanation.strip(),
            "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_annotation(NOVELTY_OUT, record)
        st.success(f"Novelty annotation saved for {pid}!")

        # Advance to next unannotated if available
        next_unannotated = [i for i in range(current_idx + 1, total_pairs) if pairs[i]["blind_pair_id"] not in load_existing_annotations(NOVELTY_OUT, annotator_id)]
        if next_unannotated:
            st.session_state["nov_pair_idx"] = next_unannotated[0]
        elif current_idx < total_pairs - 1:
            st.session_state["nov_pair_idx"] += 1
        st.rerun()


if __name__ == "__main__":
    main()
