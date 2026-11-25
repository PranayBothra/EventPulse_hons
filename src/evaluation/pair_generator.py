"""Generates blinded manual validation dataset and separate model predictions CSV."""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.clustering.similarity import MultiSignalSimilarity
from src.entities.extractor import NamedEntityExtractor
from src.representation.article_representation import (
    ArticleRepresentation,
    build_article_representation,
)
from src.representation.cache import EmbeddingCacheManager
from src.representation.embedding import BaseEmbeddingProvider
from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.pair_generator")

BLIND_ANNOTATION_INSTRUCTIONS = """# EventPulse Blind Manual Validation Dataset
# =========================================================================
# INSTRUCTIONS FOR HUMAN REVIEWERS:
# -------------------------------------------------------------------------
# You are evaluating the real-world event relationship between Article A and Article B.
# This evaluation is strictly BLIND: all model predictions, similarity scores, and
# cluster assignments have been removed to prevent anchoring bias.
#
# For each row, record exactly ONE of the following standardized labels in 'Human label':
#   - SAME_EVENT                  : Both articles report on the exact same real-world event or its immediate updates.
#   - DIFFERENT_EVENT             : The articles report on completely unrelated events or topics.
#   - SAME_TOPIC_DIFFERENT_EVENT  : Both articles share a prominent topic, entity, or actor (e.g. SpaceX, Trump, Nvidia),
#                                   but describe distinct, separate real-world occurrences.
#   - INSUFFICIENT_INFORMATION    : The headlines/summaries are too sparse or ambiguous to determine the relationship.
#
# Use 'Reviewer notes' to document justification, edge-case rationale, or observed ambiguities.
# Do NOT modify the 'Pair ID' or article identifier columns.
# =========================================================================
"""


def generate_validation_pairs_csv(
    repository: SQLiteRepository,
    embedding_provider: BaseEmbeddingProvider,
    output_path: str = "data/validation_pairs_blind.csv",
    predictions_path: Optional[str] = None,
    sample_pairs_count: int = 250,
) -> Tuple[Path, Path]:
    """Generate blinded validation CSV for reviewers and separate model predictions CSV.

    Includes diverse relationship tiers:
    - Intra-cluster pairs (predicted SAME_EVENT)
    - Hard negatives sharing entity/topic (predicted SAME_TOPIC_DIFFERENT_EVENT)
    - Unrelated cross-topic pairs (predicted DIFFERENT_EVENT)

    Returns:
        Tuple of (blind_csv_path, predictions_csv_path)
    """
    blind_file = Path(output_path)
    blind_file.parent.mkdir(parents=True, exist_ok=True)

    if predictions_path is None:
        # Default sibling file
        pred_file = blind_file.parent / (blind_file.stem.replace("_blind", "") + "_predictions.csv")
    else:
        pred_file = Path(predictions_path)
        pred_file.parent.mkdir(parents=True, exist_ok=True)

    sim_eval = MultiSignalSimilarity()
    cache_manager = EmbeddingCacheManager(repository)
    entity_extractor = NamedEntityExtractor(use_spacy=True)

    # 1. Fetch live articles from DB
    live_articles = repository.get_articles(limit=2000, is_sample=False)
    article_reps: Dict[int, ArticleRepresentation] = {}
    articles_by_id: Dict[int, Dict[str, Any]] = {}
    for a in live_articles:
        rep = build_article_representation(
            article_dict=a,
            embedding_provider=embedding_provider,
            cache_manager=cache_manager,
            entity_extractor=entity_extractor,
        )
        article_reps[a["id"]] = rep
        articles_by_id[a["id"]] = a

    # Find existing event assignments
    event_mappings: Dict[int, str] = {}
    for a in live_articles:
        ev = repository.get_event_for_article(a["id"])
        if ev:
            event_mappings[a["id"]] = ev["id"]

    candidate_pairs: List[Dict[str, Any]] = []

    # A. Sample intra-cluster pairs (predicted SAME_EVENT)
    all_events = repository.get_all_events(limit=200)
    for ev in all_events:
        if ev["article_count"] >= 2:
            members = repository.get_event_articles(ev["id"])
            for i in range(len(members)):
                for j in range(i + 1, min(i + 3, len(members))):
                    a1 = members[i]
                    a2 = members[j]
                    if a1["id"] in article_reps and a2["id"] in article_reps:
                        r1 = article_reps[a1["id"]]
                        r2 = article_reps[a2["id"]]
                        score, bd = sim_eval.compute_article_pair_similarity(r1, r2)
                        candidate_pairs.append(
                            {
                                "id_a": a1["id"],
                                "hl_a": a1["headline"],
                                "src_a": a1["source_id"],
                                "desc_a": a1.get("description", ""),
                                "id_b": a2["id"],
                                "hl_b": a2["headline"],
                                "src_b": a2["source_id"],
                                "desc_b": a2.get("description", ""),
                                "score": score,
                                "predicted": "SAME_EVENT",
                                "breakdown": bd,
                            }
                        )

    # B. Sample hard negatives: same entity or shared tokens, different events
    all_article_list = list(live_articles)
    for i in range(min(len(all_article_list), 60)):
        for j in range(i + 1, min(len(all_article_list), 60)):
            a1 = all_article_list[i]
            a2 = all_article_list[j]
            ev1 = event_mappings.get(a1["id"])
            ev2 = event_mappings.get(a2["id"])

            if ev1 and ev2 and ev1 != ev2:
                r1 = article_reps.get(a1["id"])
                r2 = article_reps.get(a2["id"])
                if r1 and r2:
                    shared_ents = r1.normalized_entities.intersection(r2.normalized_entities)
                    if shared_ents:
                        score, bd = sim_eval.compute_article_pair_similarity(r1, r2)
                        candidate_pairs.append(
                            {
                                "id_a": a1["id"],
                                "hl_a": a1["headline"],
                                "src_a": a1["source_id"],
                                "desc_a": a1.get("description", ""),
                                "id_b": a2["id"],
                                "hl_b": a2["headline"],
                                "src_b": a2["source_id"],
                                "desc_b": a2.get("description", ""),
                                "score": score,
                                "predicted": "SAME_TOPIC_DIFFERENT_EVENT",
                                "breakdown": bd,
                            }
                        )

    # C. Sample unrelated pairs
    for i in range(0, min(len(all_article_list), 40), 4):
        for j in range(i + 7, min(len(all_article_list), 40), 7):
            a1 = all_article_list[i]
            a2 = all_article_list[j]
            ev1 = event_mappings.get(a1["id"])
            ev2 = event_mappings.get(a2["id"])
            if ev1 != ev2:
                r1 = article_reps.get(a1["id"])
                r2 = article_reps.get(a2["id"])
                if r1 and r2:
                    score, bd = sim_eval.compute_article_pair_similarity(r1, r2)
                    candidate_pairs.append(
                        {
                            "id_a": a1["id"],
                            "hl_a": a1["headline"],
                            "src_a": a1["source_id"],
                            "desc_a": a1.get("description", ""),
                            "id_b": a2["id"],
                            "hl_b": a2["headline"],
                            "src_b": a2["source_id"],
                            "desc_b": a2.get("description", ""),
                            "score": score,
                            "predicted": "DIFFERENT_EVENT",
                            "breakdown": bd,
                        }
                    )

    # Deduplicate candidate pairs
    unique_pairs = []
    seen = set()
    for p in candidate_pairs:
        pair_key = tuple(sorted([p["id_a"], p["id_b"]]))
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append(p)

    # Select target count
    selected_pairs = unique_pairs[:sample_pairs_count]

    # Assign consistent pair IDs (pair_001, pair_002, ...)
    blind_rows = []
    pred_rows = []
    for idx, p in enumerate(selected_pairs, 1):
        pair_id = f"pair_{idx:03d}"

        # Blinded row for human reviewer (NO predictions, NO scores)
        blind_rows.append(
            {
                "Pair ID": pair_id,
                "Article ID A": p["id_a"],
                "Headline A": p["hl_a"],
                "Source A": p["src_a"],
                "Description A": p["desc_a"],
                "Article ID B": p["id_b"],
                "Headline B": p["hl_b"],
                "Source B": p["src_b"],
                "Description B": p["desc_b"],
                "Human label": "PENDING_ANNOTATIONS",
                "Reviewer notes": "",
            }
        )

        # Separate predictions row
        bd = p.get("breakdown", {})
        pred_rows.append(
            {
                "Pair ID": pair_id,
                "Article ID A": p["id_a"],
                "Article ID B": p["id_b"],
                "Predicted similarity": f"{p['score']:.4f}",
                "Predicted assignment": p["predicted"],
                "Semantic score": f"{bd.get('semantic', 0.0):.4f}",
                "Entity score": f"{bd.get('entity', 0.0):.4f}",
                "Headline score": f"{bd.get('headline', 0.0):.4f}",
                "Temporal score": f"{bd.get('temporal', 0.0):.4f}",
            }
        )

    # Write blind CSV
    blind_fields = [
        "Pair ID",
        "Article ID A",
        "Headline A",
        "Source A",
        "Description A",
        "Article ID B",
        "Headline B",
        "Source B",
        "Description B",
        "Human label",
        "Reviewer notes",
    ]
    with open(blind_file, "w", encoding="utf-8", newline="") as f:
        f.write(BLIND_ANNOTATION_INSTRUCTIONS)
        writer = csv.DictWriter(f, fieldnames=blind_fields)
        writer.writeheader()
        for r in blind_rows:
            writer.writerow(r)

    # Write predictions CSV
    pred_fields = [
        "Pair ID",
        "Article ID A",
        "Article ID B",
        "Predicted similarity",
        "Predicted assignment",
        "Semantic score",
        "Entity score",
        "Headline score",
        "Temporal score",
    ]
    with open(pred_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        for r in pred_rows:
            writer.writerow(r)

    logger.info("Generated %d blinded pairs at %s and predictions at %s", len(blind_rows), blind_file, pred_file)
    return blind_file, pred_file
