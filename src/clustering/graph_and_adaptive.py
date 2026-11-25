"""Stage 3.10 Clustering Redesign:
1. GCom-inspired Sentence-Level Similarity-Graph + Community Detection baseline
   (adapted from Nikolaidis et al., EMNLP 2026 Findings, using cached MiniLM/multilingual
   sentence embeddings + NetworkX Louvain community detection on sparse top-k graphs).
2. Adaptive EventPulse Incremental Clustering (semantic-primary without sparse-signal drag
   or hard 168h timestamp rejection).
3. Adaptive EventPulse Graph-Assisted Clustering (sentence graph + entity/temporal modulation).
"""

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.novelty.sentence_extraction import SentenceExtractor


BOILERPLATE_TITLE_RE = re.compile(
    r"^(watch|video|live updates?|photos?|gallery|podcast|breaking)\b|\|\s*[a-z0-9.-]+$|\bsubscribe\b|\ball rights reserved\b",
    re.I,
)


def acronym_set(entities: Set[str]) -> Set[str]:
    """Expand entities with simple multi-word acronyms for alias resilience."""
    out = set(entities)
    for ent in entities:
        parts = [p for p in re.split(r"[\s\-]+", ent) if p and p[0].isalpha()]
        if len(parts) >= 2:
            acr = "".join(p[0].lower() for p in parts)
            if len(acr) >= 2:
                out.add(acr)
    return out


def compute_alias_aware_entity_sim(ents_a: Set[str], ents_b: Set[str]) -> float:
    """Jaccard overlap with simple acronym expansion; returns 0.0 (not 0.5 penalty/boost) when empty."""
    if not ents_a or not ents_b:
        return 0.0
    ea = acronym_set({e.lower() for e in ents_a})
    eb = acronym_set({e.lower() for e in ents_b})
    u = len(ea | eb)
    return len(ea & eb) / u if u > 0 else 0.0


def extract_informative_article_sentences(
    articles: List[Dict[str, Any]],
    extractor: Optional[SentenceExtractor] = None,
    max_sents_per_doc: int = 8,
    min_info_score: float = 0.25,
    max_df_ratio: float = 0.03,
) -> Tuple[List[List[int]], List[str], Dict[str, Any]]:
    """Extract informative, non-boilerplate sentence pools with corpus-level boilerplate pruning.

    Deduplicates unique sentence strings and drops hyper-frequent generic sentences
    appearing in > max_df_ratio of articles (minimum count threshold >= 5) so syndicated
    footers/headers cannot bridge unrelated stories in the similarity graph.
    """
    ext = extractor or SentenceExtractor()
    raw_doc_sents: List[List[str]] = []
    sent_df: Counter = Counter()

    for a in articles:
        title = (a.get("title") or a.get("headline") or "").strip()
        body = (a.get("text") or a.get("description") or "").strip()
        extracted = ext.extract_from_article(headline=title, body=body[:2500])

        kept: List[str] = []
        for ex in extracted:
            st = ex.sentence_text
            if ex.is_headline and not BOILERPLATE_TITLE_RE.search(st):
                kept.append(st)
            elif ex.informativeness_score >= min_info_score and not BOILERPLATE_TITLE_RE.search(st):
                kept.append(st)
            if len(kept) >= max_sents_per_doc:
                break

        if not kept:
            fallback = (title or body[:240]).strip()
            if fallback:
                kept = [fallback]

        unique_doc_sents = list(dict.fromkeys(kept))
        raw_doc_sents.append(unique_doc_sents)
        for s in unique_doc_sents:
            sent_df[s.lower()] += 1

    n_docs = len(articles)
    df_cap = max(5, int(math.ceil(max_df_ratio * max(1, n_docs))))

    unique_sents: List[str] = []
    sent_to_uid: Dict[str, int] = {}
    doc_sent_uids: List[List[int]] = []
    dropped_generic = 0

    for idx, sents in enumerate(raw_doc_sents):
        uids: List[int] = []
        for s in sents:
            if sent_df[s.lower()] > df_cap and len(sents) > 1:
                dropped_generic += 1
                continue
            if s not in sent_to_uid:
                sent_to_uid[s] = len(unique_sents)
                unique_sents.append(s)
            uids.append(sent_to_uid[s])
        if not uids and sents:
            s0 = sents[0]
            if s0 not in sent_to_uid:
                sent_to_uid[s0] = len(unique_sents)
                unique_sents.append(s0)
            uids.append(sent_to_uid[s0])
        doc_sent_uids.append(uids)

    stats = {
        "n_docs": n_docs,
        "unique_informative_sentences": len(unique_sents),
        "mean_sents_per_doc": round(float(np.mean([len(u) for u in doc_sent_uids])), 2) if doc_sent_uids else 0.0,
        "dropped_generic_sentences": dropped_generic,
        "df_cap": df_cap,
    }
    return doc_sent_uids, unique_sents, stats


def build_sentence_document_graph(
    n_docs: int,
    doc_sent_uids: List[List[int]],
    sent_vecs: np.ndarray,
    doc_vecs: Optional[np.ndarray] = None,
    k_sent_neighbors: int = 20,
    sent_sim_threshold: float = 0.60,
    top_doc_neighbors: int = 25,
    doc_edge_threshold: float = 0.45,
    sent_weight: float = 0.65,
    doc_weight: float = 0.35,
    entities: Optional[List[Set[str]]] = None,
    headline_tokens: Optional[List[Set[str]]] = None,
    epochs: Optional[List[Optional[float]]] = None,
    entity_boost: float = 0.0,
    headline_boost: float = 0.0,
    temporal_penalty_weight: float = 0.0,
    soft_temporal_hours: float = 96.0,
    chunk_size: int = 512,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Construct sparse document similarity graph from top-k sentence interactions (+ optional metadata modulation).

    Memory-bounded: computes sentence-pair dot products in (chunk_size x M) float32 blocks
    without materializing an N x N or M x M dense similarity matrix.
    """
    uid_to_docs: Dict[int, List[int]] = defaultdict(list)
    for d_idx, uids in enumerate(doc_sent_uids):
        for u in uids:
            uid_to_docs[u].append(d_idx)

    # Accumulate strongest sentence-link weights per document pair (i < j)
    pair_sent_sims: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    n_sents = sent_vecs.shape[0]
    k_eff = min(k_sent_neighbors + 1, n_sents)

    for start in range(0, n_sents, chunk_size):
        end = min(start + chunk_size, n_sents)
        block = sent_vecs[start:end] @ sent_vecs.T  # (chunk, M) float32
        # Top-k indices along axis 1
        top_idx = np.argpartition(-block, kth=k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(end - start)[:, None]
        top_vals = block[rows, top_idx]

        for r_off in range(end - start):
            u = start + r_off
            docs_u = uid_to_docs[u]
            for c_pos in range(k_eff):
                v = int(top_idx[r_off, c_pos])
                if v == u:
                    continue
                sim = float(top_vals[r_off, c_pos])
                if sim < sent_sim_threshold:
                    continue
                docs_v = uid_to_docs[v]
                for du in docs_u:
                    for dv in docs_v:
                        if du == dv:
                            continue
                        key = (du, dv) if du < dv else (dv, du)
                        pair_sent_sims[key].append(sim)

    # Optionally include sparse top-k document-level cosine candidates so headline/body parity works seamlessly
    pair_doc_sims: Dict[Tuple[int, int], float] = {}
    if doc_vecs is not None:
        k_doc_eff = min(top_doc_neighbors + 1, n_docs)
        for start in range(0, n_docs, chunk_size):
            end = min(start + chunk_size, n_docs)
            d_block = doc_vecs[start:end] @ doc_vecs.T
            top_d_idx = np.argpartition(-d_block, kth=k_doc_eff - 1, axis=1)[:, :k_doc_eff]
            rows = np.arange(end - start)[:, None]
            top_d_vals = d_block[rows, top_d_idx]
            for r_off in range(end - start):
                di = start + r_off
                for c_pos in range(k_doc_eff):
                    dj = int(top_d_idx[r_off, c_pos])
                    if dj == di:
                        continue
                    dsim = float(top_d_vals[r_off, c_pos])
                    if dsim >= max(0.35, doc_edge_threshold - 0.15):
                        key = (di, dj) if di < dj else (dj, di)
                        if dsim > pair_doc_sims.get(key, 0.0):
                            pair_doc_sims[key] = dsim

    # Aggregate final weighted sparse edges
    candidate_edges = set(pair_sent_sims.keys()) | set(pair_doc_sims.keys())
    G = nx.Graph()
    G.add_nodes_from(range(n_docs))

    # Sparse neighbor retention: keep at most top-25 strongest incident edges per document
    exp_ents = [acronym_set({e.lower() for e in s}) for s in entities] if (entities is not None and entity_boost > 0.0) else None
    max_deg = 25
    per_node_edges: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for key in candidate_edges:
        i, j = key
        s_list = pair_sent_sims.get(key)
        if s_list:
            # Top-2 strongest sentence interactions dampened slightly so 1 noisy link < 2 coherent links
            s_sorted = sorted(s_list, reverse=True)[:2]
            s_score = float(np.mean(s_sorted))
        else:
            s_score = 0.0

        if key in pair_doc_sims:
            d_score = pair_doc_sims[key]
        elif doc_vecs is not None:
            d_score = float(np.clip(doc_vecs[i] @ doc_vecs[j], 0.0, 1.0))
        else:
            d_score = s_score

        if doc_vecs is not None:
            base = sent_weight * s_score + doc_weight * d_score if s_score > 0 else d_score * 0.90
        else:
            base = s_score

        if exp_ents is not None:
            ea, eb = exp_ents[i], exp_ents[j]
            if ea and eb:
                u = len(ea | eb)
                if u > 0:
                    base += entity_boost * (len(ea & eb) / u)

        if headline_tokens is not None and headline_boost > 0.0 and headline_tokens[i] and headline_tokens[j]:
            hu = len(headline_tokens[i] | headline_tokens[j])
            if hu > 0:
                base += headline_boost * (len(headline_tokens[i] & headline_tokens[j]) / hu)

        if epochs is not None and temporal_penalty_weight > 0.0:
            ei, ej = epochs[i], epochs[j]
            if ei is not None and ej is not None:
                diff_h = abs(ei - ej) / 3600.0
                if diff_h > soft_temporal_hours:
                    # Smooth bounded penalty; never blocks very strong semantic evidence (>= 0.78)
                    if max(s_score, d_score) < 0.78:
                        pen = 1.0 - math.exp(-0.01 * (diff_h - soft_temporal_hours))
                        base -= temporal_penalty_weight * pen

        if base >= doc_edge_threshold:
            w = float(base)
            per_node_edges[i].append((w, j))
            per_node_edges[j].append((w, i))

    for u, nbrs in per_node_edges.items():
        if len(nbrs) > max_deg:
            nbrs.sort(key=lambda x: -x[0])
            nbrs = nbrs[:max_deg]
        for w, v in nbrs:
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=w)

    edges_added = G.number_of_edges()

    g_stats = {
        "nodes": n_docs,
        "edges": edges_added,
        "avg_degree": round(2.0 * edges_added / max(1, n_docs), 2),
        "connected_components": nx.number_connected_components(G),
    }
    return G, g_stats


def detect_communities_louvain(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42,
) -> List[int]:
    """Partition sparse document graph via Louvain modularity per connected component."""
    n_docs = G.number_of_nodes()
    labels = [-1] * n_docs
    next_cid = 0
    for comp in nx.connected_components(G):
        if len(comp) == 1:
            u = next(iter(comp))
            labels[u] = next_cid
            next_cid += 1
        elif len(comp) == 2:
            for u in comp:
                labels[u] = next_cid
            next_cid += 1
        else:
            subG = G.subgraph(comp)
            comms = nx.community.louvain_communities(subG, weight="weight", resolution=resolution, seed=seed)
            for comm in comms:
                for node in comm:
                    labels[node] = next_cid
                next_cid += 1
    return labels


def run_adaptive_incremental_clustering(
    n_docs: int,
    vecs: np.ndarray,
    entities: List[Set[str]],
    headline_tokens: List[Set[str]],
    epochs: List[Optional[float]],
    match_threshold: float = 0.50,
    entity_bonus: float = 0.08,
    headline_bonus: float = 0.04,
    temporal_bonus: float = 0.03,
    temporal_penalty: float = 0.04,
    soft_temporal_hours: float = 96.0,
    semantic_high_bypass: float = 0.65,
) -> List[int]:
    """Semantic-primary incremental clustering without sparse-signal drag or hard temporal cutoffs.

    Design principles (Stage 3.10):
    - Dense semantic cosine similarity against running L2 centroid is primary evidence.
    - Missing entities, differing headlines, or missing/distant dates do NOT apply punitive
      weights that drag genuine same-event pairs below threshold.
    - Entity consistency (alias-aware) and headline overlap act as bounded supporting evidence.
    - Temporal distance provides a tiny freshness bonus when aligned and a bounded soft penalty
      when far apart (> 96h), bypassed whenever semantic cosine >= semantic_high_bypass.
    """
    decay_lambda = math.log(2.0) / 72.0
    exp_ents = [acronym_set({e.lower() for e in s}) for s in entities]
    d = vecs.shape[1]
    cluster_sums = np.empty((n_docs, d), dtype=np.float32)
    cluster_centroids = np.empty((n_docs, d), dtype=np.float32)
    n_clusters = 0
    cluster_ents: List[Set[str]] = []
    cluster_heads: List[Set[str]] = []
    cluster_epochs: List[Optional[float]] = []

    pred_labels = [-1] * n_docs

    for i in range(n_docs):
        v = vecs[i]
        e_set = exp_ents[i]
        h_set = headline_tokens[i]
        ep = epochs[i]

        if n_clusters == 0:
            pred_labels[i] = 0
            cluster_sums[0] = v
            cluster_centroids[0] = v
            n_clusters = 1
            cluster_ents.append(set(e_set))
            cluster_heads.append(set(h_set))
            cluster_epochs.append(ep)
            continue

        sem_sims = np.clip(cluster_centroids[:n_clusters] @ v, 0.0, 1.0)

        # Max possible bonus is entity_bonus + headline_bonus + temporal_bonus
        max_bonus = entity_bonus + headline_bonus + temporal_bonus
        min_sem = max(0.0, match_threshold - max_bonus)
        cand_cids = np.where(sem_sims >= min_sem)[0]
        if len(cand_cids) > 40:
            cand_cids = cand_cids[np.argsort(-sem_sims[cand_cids])[:40]]

        best_cid = -1
        best_score = -1.0
        for cid in cand_cids:
            s_sem = float(sem_sims[cid])
            score = s_sem
            if entity_bonus > 0.0 and e_set and cluster_ents[cid]:
                u = len(e_set | cluster_ents[cid])
                if u > 0:
                    score += entity_bonus * (len(e_set & cluster_ents[cid]) / u)
            if headline_bonus > 0.0 and h_set and cluster_heads[cid]:
                hu = len(h_set | cluster_heads[cid])
                if hu > 0:
                    score += headline_bonus * (len(h_set & cluster_heads[cid]) / hu)

            c_ep = cluster_epochs[cid]
            if ep is not None and c_ep is not None:
                diff_h = abs(ep - c_ep) / 3600.0
                if diff_h <= soft_temporal_hours:
                    score += temporal_bonus * math.exp(-decay_lambda * diff_h)
                elif s_sem < semantic_high_bypass and temporal_penalty > 0.0:
                    pen = 1.0 - math.exp(-0.01 * (diff_h - soft_temporal_hours))
                    score -= temporal_penalty * pen

            if score > best_score:
                best_score = score
                best_cid = int(cid)

        if best_cid >= 0 and best_score >= match_threshold:
            pred_labels[i] = best_cid
            cluster_sums[best_cid] += v
            norm = float(np.linalg.norm(cluster_sums[best_cid]))
            cluster_centroids[best_cid] = cluster_sums[best_cid] / norm if norm > 1e-12 else v
            cluster_ents[best_cid].update(e_set)
            cluster_heads[best_cid].update(h_set)
            if ep is not None:
                if cluster_epochs[best_cid] is None or ep > cluster_epochs[best_cid]:
                    cluster_epochs[best_cid] = ep
        else:
            new_cid = n_clusters
            pred_labels[i] = new_cid
            cluster_sums[new_cid] = v
            cluster_centroids[new_cid] = v
            n_clusters += 1
            cluster_ents.append(set(e_set))
            cluster_heads.append(set(h_set))
            cluster_epochs.append(ep)

    return pred_labels
