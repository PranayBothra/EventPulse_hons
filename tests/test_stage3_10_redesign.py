"""Stage 3.10 unit tests covering redesigned clustering & proposition-aware novelty logic."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clustering.graph_and_adaptive import (
    compute_alias_aware_entity_sim,
    extract_informative_article_sentences,
    run_adaptive_incremental_clustering,
)
from src.novelty.proposition_estimator import PropositionAnalyzer, RedesignedNoveltyEvaluator, compare_propositions


def test_alias_aware_entity_sim_matches_acronym():
    sim = compute_alias_aware_entity_sim({"European Union", "NATO"}, {"EU", "NATO"})
    assert sim > 0.5


def test_generic_sentence_pruning_prevents_boilerplate_bridges():
    # Repeated identical syndicated sentence appearing across > df_cap (5) distinct articles must be DF-pruned
    articles = [
        {
            "title": f"Unique event headline {i} at Regional Airport today",
            "text": "Additional reporting was contributed by international wire bureaus across regional offices. Local emergency responders handled the incident.",
        }
        for i in range(7)
    ]
    uids, unique_sents, stats = extract_informative_article_sentences(articles, max_df_ratio=0.30)
    assert stats["dropped_generic_sentences"] > 0
    assert not any("international wire bureaus" in s for s in unique_sents)


def test_adaptive_incremental_allows_high_semantic_despite_missing_time_and_entities():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.92, 0.39, 0.0], dtype=np.float32)
    v2 /= np.linalg.norm(v2)
    v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    preds = run_adaptive_incremental_clustering(
        n_docs=3,
        vecs=np.vstack([v1, v2, v3]),
        entities=[set(), set(), {"Unrelated"}],
        headline_tokens=[{"a"}, {"b"}, {"c"}],
        epochs=[1700000000.0, None, 1700000000.0 + 30 * 86400.0],
        match_threshold=0.52,
    )
    assert preds[0] == preds[1]
    assert preds[2] != preds[0]


def test_proposition_numerical_attribute_alignment_and_role_swap():
    analyzer = PropositionAnalyzer()

    # 1. Compatible numerical attribute update -> matched_num_delta == 1.0
    p_mem_num = analyzer.extract("Authorities reported 38 casualties in the tunnel collapse.")
    p_cand_num = analyzer.extract("Authorities reported 47 casualties in the tunnel collapse.")
    feats_match = compare_propositions(p_cand_num, p_mem_num, cos_sim=0.90)
    assert feats_match["matched_num_delta"] == 1.0

    # 2. Unrelated attributes ("38 million" vs "47 schools") -> must NOT fire matched_num_delta
    p_proj = analyzer.extract("The municipal project costs $38 million.")
    p_sch = analyzer.extract("The city has 47 schools.")
    feats_unrel = compare_propositions(p_sch, p_proj, cos_sim=0.30)
    assert feats_unrel["matched_num_delta"] == 0.0

    # 3. Subject-object role swap ("Police arrested suspect" vs "Suspect arrested police")
    p_pol = analyzer.extract("Police arrested the suspect.")
    p_swp = analyzer.extract("The suspect arrested the police.")
    feats_swp = compare_propositions(p_swp, p_pol, cos_sim=0.95)
    assert feats_swp["role_reversal"] == 1.0

    # 4. Passive voice equivalence must NOT trigger role reversal
    p_act = analyzer.extract("Federal prosecutors indicted three executives on fraud charges.")
    p_pas = analyzer.extract("Three executives were indicted by federal prosecutors on fraud charges.")
    feats_voice = compare_propositions(p_pas, p_act, cos_sim=0.95)
    assert feats_voice["role_reversal"] == 0.0


def test_emerging_topic_separates_new_cluster_from_absorption():
    from src.novelty.emerging_topic import EventPulseClusterEmergenceTracker, RollingWindowEmergenceDetector

    ep = EventPulseClusterEmergenceTracker(match_threshold=0.60)
    rw = RollingWindowEmergenceDetector(cur_window=10, prev_window=60, cluster_sim=0.60, history_sim_cutoff=0.58)

    # Seed 60 orthogonal-ish real background vectors before onset=60
    rng = np.random.default_rng(7)
    bg_vec = np.zeros(8, dtype=np.float32)
    bg_vec[0] = 1.0
    for step in range(60):
        v = rng.normal(0, 0.05, size=8).astype(np.float32)
        v[step % 4] += 1.0
        v /= np.linalg.norm(v)
        ep.process_one(step, f"r_{step}", v, False, None, onset_step=60)
        rw.process_one(step, f"r_{step}", v, False, None, onset_step=60)

    # 1. Inject novel topic along untouched dimension 7 -> MUST form a genuinely new cluster
    nov = np.zeros(8, dtype=np.float32)
    nov[7] = 1.0
    out_ep_new = ep.process_one(60, "syn_1", nov, True, "scen_new", onset_step=60)
    out_rw_new = rw.process_one(60, "syn_1", nov, True, "scen_new", onset_step=60)
    assert out_ep_new["mode"] == "new_cluster_formation"
    assert not out_ep_new["seeded_before_onset"]
    assert out_rw_new["mode"] == "window_emergence"

    # 2. Inject item closely aligned to pre-onset cluster -> MUST be flagged as absorption
    abs_vec = np.zeros(8, dtype=np.float32)
    abs_vec[0] = 1.0
    out_ep_abs = ep.process_one(61, "syn_2", abs_vec, True, "scen_abs", onset_step=60)
    assert out_ep_abs["seeded_before_onset"] is True
    assert out_ep_abs["mode"] == "absorption_into_existing"
