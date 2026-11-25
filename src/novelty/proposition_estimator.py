"""Stage 3.10 Proposition-Aware Novelty Estimator & Conditional Evidence Extractor.

Replaces unconditional hard heuristic overrides (max(0.80, base) / max(0.85, base))
with dependency-parsed Subject-Predicate-Object roles, attribute-aligned numerical
deltas, conditional polarity/action reversals, epistemic modality shifts, and
clean features suitable for regularized calibrated fusion.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.entities.extractor import NamedEntityExtractor
from src.novelty.multi_signal_estimator import NEGATION_TOKENS, OPPOSITION_PAIRS, PROMOTIONAL_PATTERNS
from src.novelty.sentence_filtering import SentenceFilter


WORD_UNITS: Dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALE_WORDS: Dict[str, float] = {
    "hundred": 1e2,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}

# Semantic groups for attribute compatibility when wording slightly varies
ATTRIBUTE_SYNONYM_GROUPS: Dict[str, str] = {
    "casualty": "casualties", "killed": "casualties", "dead": "casualties", "death": "casualties",
    "injured": "casualties", "wounded": "casualties", "hurt": "casualties", "victim": "casualties",
    "fatality": "casualties",
    "people": "count_person", "person": "count_person", "resident": "count_person",
    "passenger": "count_person", "worker": "count_person", "employee": "count_person",
    "household": "count_household", "home": "count_household", "family": "count_household",
    "aircraft": "count_vehicle", "plane": "count_vehicle", "jet": "count_vehicle",
    "coach": "count_vehicle", "car": "count_vehicle", "train": "count_vehicle",
    "school": "count_facility", "hospital": "count_facility", "building": "count_facility",
    "dollar": "currency", "usd": "currency", "$": "currency", "euro": "currency", "€": "currency",
    "pound": "currency", "£": "currency", "aid": "currency", "cost": "currency", "fine": "currency",
    "percent": "percentage", "%": "percentage", "rate": "percentage", "inflation": "percentage",
    "kilometer": "distance", "kilometre": "distance", "km": "distance", "mile": "distance",
    "vote": "vote", "seat": "vote",
    "position": "count_job", "job": "count_job", "role": "count_job",
    "ton": "mass", "metric": "mass", "kg": "mass",
    "day": "duration", "hour": "duration", "month": "duration", "year": "duration", "week": "duration",
}

LEMMA_OPPOSITION_PAIRS: List[Tuple[Set[str], Set[str]]] = [
    ({"approve", "pass", "ratify", "accept", "grant", "confirm", "uphold"}, {"reject", "veto", "deny", "block", "overturn", "refuse"}),
    ({"arrest", "detain", "charge", "indict", "convict"}, {"release", "acquit", "free", "clear", "drop"}),
    ({"open", "reopen", "resume", "launch", "start", "begin"}, {"close", "shut", "suspend", "halt", "cancel", "end", "stop"}),
    ({"increase", "rise", "climb", "surge", "jump", "expand", "gain"}, {"decrease", "fall", "drop", "plummet", "shrink", "cut", "lose"}),
    ({"win", "defeat"}, {"lose"}),
    ({"impose", "enact"}, {"lift", "repeal"}),
    ({"sign", "agree"}, {"withdraw", "collapse"}),
]

HEDGED_MODALITY_RE = re.compile(
    r"\b(may|might|could|would|possible|possibly|potential|potentially|proposed|draft|"
    r"suspect|suspected|consider|weigh|debate|warned|expected|likely|scheduled|plan|planned)\b",
    re.I,
)
CONFIRMED_ACTION_RE = re.compile(
    r"\b(enacted|formally|officially|signed|confirmed|certified|eliminated|rescued|located|"
    r"caused|collapsed|imposed|passed|ratified|acquitted|convicted|killed|completed)\b",
    re.I,
)
EXTENDED_BOILERPLATE_PATTERNS = list(PROMOTIONAL_PATTERNS) + [
    re.compile(r"\b(follow us on social media|third[- ]party advertisement|read our privacy policy|breaking news newsletter)\b", re.I),
]


@dataclass
class QuantityMeasure:
    value: float
    unit_class: str  # 'currency', 'percentage', 'distance', 'duration', 'mass', 'count', etc.
    attr_group: str  # normalized head noun/attribute group


@dataclass
class ExtractedProposition:
    text: str
    entities: Set[str] = field(default_factory=set)
    subjects: Set[str] = field(default_factory=set)      # Logical agent / actor lemmas
    objects: Set[str] = field(default_factory=set)       # Logical patient / target lemmas
    predicates: Set[str] = field(default_factory=set)    # Predicate verb lemmas
    negated: bool = False
    hedged_modality: bool = False
    confirmed_modality: bool = False
    quantities: List[QuantityMeasure] = field(default_factory=list)
    temporal_terms: Set[str] = field(default_factory=set)
    content_lemmas: Set[str] = field(default_factory=set)
    is_boilerplate: bool = False


def _parse_number_words_phrase(tokens: List[str]) -> Optional[float]:
    total = 0.0
    current = 0.0
    matched = False
    for w in tokens:
        wl = w.lower().strip("-")
        if wl in WORD_UNITS:
            current += WORD_UNITS[wl]
            matched = True
        elif wl == "hundred":
            current = max(1.0, current) * 100.0
            matched = True
        elif wl in SCALE_WORDS:
            scale = SCALE_WORDS[wl]
            total += max(1.0, current) * scale
            current = 0.0
            matched = True
        else:
            break
    if not matched:
        return None
    return total + current


def _is_calendar_year_or_code(val: float, head_lemma: str, raw_tok: str) -> bool:
    """Filter out train route codes (59320), zip codes, or bare 4-digit calendar years unless explicitly quantified."""
    if head_lemma in ("train", "flight", "route", "highway", "bus", "section", "article"):
        return True
    if 1900 <= val <= 2035 and raw_tok.isdigit() and len(raw_tok) == 4 and head_lemma in ("", "year", "in", "since", "during"):
        return False  # year context can be checked via temporal expression, don't treat as count!
    if val >= 10000 and "." not in raw_tok and head_lemma in ("express", "passenger", "train", "flight"):
        return True
    return False


class PropositionAnalyzer:
    """Lightweight CPU-friendly dependency & proposition extractor reusing spaCy en_core_web_sm."""

    def __init__(self, ner: Optional[NamedEntityExtractor] = None):
        self.ner = ner or NamedEntityExtractor(use_spacy=True)
        self.nlp = self.ner.nlp

    @staticmethod
    def is_boilerplate(text: str) -> bool:
        return any(p.search(text) for p in EXTENDED_BOILERPLATE_PATTERNS)

    def extract(self, text: str) -> ExtractedProposition:
        prop = ExtractedProposition(text=text)
        prop.is_boilerplate = self.is_boilerplate(text)
        prop.hedged_modality = bool(HEDGED_MODALITY_RE.search(text))
        prop.confirmed_modality = bool(CONFIRMED_ACTION_RE.search(text))

        if self.nlp is None:
            prop.content_lemmas = {w.lower() for w in re.findall(r"\b[a-z]{3,}\b", text)}
            prop.negated = bool(set(re.findall(r"\b\w+\b", text.lower())) & NEGATION_TOKENS) or "not guilty" in text.lower()
            return prop

        doc = self.nlp(text[:600])
        ents: Set[str] = set()
        for e in doc.ents:
            if e.label_ in ("DATE", "TIME"):
                prop.temporal_terms.add(e.text.lower())
            elif e.label_ in ("PERSON", "ORG", "GPE", "NORP", "LOC", "EVENT", "PRODUCT", "FAC"):
                ents.add(e.text.lower())
        prop.entities = ents

        # Extract logical SPO roles normalizing passive voice ('X was killed by Y' -> subj={Y}, obj={X})
        is_passive = any(tok.dep_ in ("nsubjpass", "auxpass") for tok in doc)
        nsubj_lemmas: Set[str] = set()
        subjpass_lemmas: Set[str] = set()
        dobj_lemmas: Set[str] = set()
        by_agent_lemmas: Set[str] = set()

        def span_tokens(head_tok) -> Set[str]:
            # Gather compound/amod noun phrase core tokens for stable entity role comparison
            out = {head_tok.lemma_.lower()}
            for ch in head_tok.children:
                if ch.dep_ in ("compound", "amod", "flat", "nmod"):
                    out.add(ch.lemma_.lower())
            # Also full subtree phrase if short
            sub = " ".join(t.text.lower() for t in head_tok.subtree if not t.is_punct and t.dep_ != "det")
            if 2 <= len(sub) <= 45:
                out.add(sub)
            return {x for x in out if len(x) >= 2 and x not in ("the", "this", "that", "which", "who")}

        for tok in doc:
            if tok.pos_ in ("VERB",) or tok.dep_ == "ROOT":
                if tok.pos_ in ("VERB", "AUX") and tok.lemma_.lower() not in ("be", "have", "do"):
                    prop.predicates.add(tok.lemma_.lower())
            if tok.dep_ == "neg" or tok.lower_ in NEGATION_TOKENS:
                prop.negated = True
            if tok.dep_ == "nsubj":
                nsubj_lemmas.update(span_tokens(tok))
            elif tok.dep_ == "nsubjpass":
                subjpass_lemmas.update(span_tokens(tok))
            elif tok.dep_ in ("dobj", "obj", "attr", "oprd"):
                dobj_lemmas.update(span_tokens(tok))
            elif tok.dep_ == "pobj":
                if tok.head.lower_ == "by" and is_passive:
                    by_agent_lemmas.update(span_tokens(tok))
                else:
                    dobj_lemmas.update(span_tokens(tok))
            if not tok.is_stop and tok.is_alpha and len(tok.lemma_) >= 3:
                prop.content_lemmas.add(tok.lemma_.lower())

        if is_passive:
            prop.subjects = by_agent_lemmas if by_agent_lemmas else nsubj_lemmas
            prop.objects = subjpass_lemmas | dobj_lemmas
        else:
            prop.subjects = nsubj_lemmas
            prop.objects = dobj_lemmas

        # Extract normalized QuantityMeasure spans (handling digits AND number words like "three hundred")
        i = 0
        toks = list(doc)
        while i < len(toks):
            t = toks[i]
            val: Optional[float] = None
            end_j = i + 1
            unit_cls = "count"
            raw_clean = t.text.replace(",", "").replace("$", "").replace("%", "")

            if re.fullmatch(r"\d+(?:\.\d+)?", raw_clean):
                val = float(raw_clean)
                if "$" in t.text or (i > 0 and toks[i - 1].text in ("$", "€", "£")):
                    unit_cls = "currency"
                elif "%" in t.text:
                    unit_cls = "percentage"
                # Check immediate scale word ("million", "billion", "percent")
                if end_j < len(toks):
                    nxt = toks[end_j].lower_
                    if nxt in SCALE_WORDS:
                        val *= SCALE_WORDS[nxt]
                        end_j += 1
                    elif nxt in ("percent", "%"):
                        unit_cls = "percentage"
                        end_j += 1
            elif t.lower_ in WORD_UNITS:
                w_seq = []
                j = i
                while j < len(toks) and (toks[j].lower_ in WORD_UNITS or toks[j].lower_ in SCALE_WORDS):
                    w_seq.append(toks[j].text)
                    j += 1
                val = _parse_number_words_phrase(w_seq)
                end_j = j

            if val is not None:
                # Inspect head noun + immediate right context nouns for semantic attribute
                head_lemma = t.head.lemma_.lower() if t.head != t else ""
                if head_lemma in ("%", "percent"):
                    unit_cls = "percentage"
                elif head_lemma in ("$", "dollar", "euro", "pound"):
                    unit_cls = "currency"

                attr_candidates = [head_lemma]
                for k in range(end_j, min(len(toks), end_j + 4)):
                    if toks[k].pos_ in ("NOUN", "PROPN", "ADJ", "VERB"):
                        attr_candidates.append(toks[k].lemma_.lower())
                if t.head.head != t.head and t.head.head.pos_ in ("NOUN", "VERB"):
                    attr_candidates.append(t.head.head.lemma_.lower())

                if not _is_calendar_year_or_code(val, head_lemma, raw_clean):
                    attr_group = "generic"
                    for ac in attr_candidates:
                        if ac in ATTRIBUTE_SYNONYM_GROUPS:
                            attr_group = ATTRIBUTE_SYNONYM_GROUPS[ac]
                            if attr_group in ("currency", "percentage", "distance", "duration", "mass"):
                                unit_cls = attr_group
                            break
                    if attr_group == "generic" and head_lemma and head_lemma not in ("be", "have", "do", "say"):
                        attr_group = head_lemma
                    prop.quantities.append(QuantityMeasure(value=val, unit_class=unit_cls, attr_group=attr_group))
                i = end_j
            else:
                i += 1

        return prop


def compare_propositions(cand: ExtractedProposition, mem: ExtractedProposition, cos_sim: float) -> Dict[str, float]:
    """Compute conditional proposition-comparison signals (zero hard overrides)."""
    shared_ents = cand.entities & mem.entities
    all_ents = cand.entities | mem.entities
    ent_jaccard = len(shared_ents) / len(all_ents) if all_ents else 0.0

    shared_lemmas = cand.content_lemmas & mem.content_lemmas
    min_lem = min(len(cand.content_lemmas), len(mem.content_lemmas))
    lemma_overlap = len(shared_lemmas) / min_lem if min_lem > 0 else 0.0

    compatible_context = (cos_sim >= 0.55) or (len(shared_ents) > 0) or (lemma_overlap >= 0.35)

    # 1. Attribute-aligned numerical update check
    matched_num_delta = 0.0
    unrelated_num_diff = 0.0
    if cand.quantities and mem.quantities:
        aligned_found = False
        unaligned_found = False
        for qc in cand.quantities:
            for qm in mem.quantities:
                same_unit = (qc.unit_class == qm.unit_class)
                same_attr = (qc.attr_group == qm.attr_group) or (
                    qc.unit_class in ("currency", "percentage") and qm.unit_class == qc.unit_class
                )
                rel_diff = abs(qc.value - qm.value) / max(1.0, abs(qc.value), abs(qm.value))
                if rel_diff > 0.01:
                    if compatible_context and same_unit and same_attr:
                        aligned_found = True
                    else:
                        unaligned_found = True
        if aligned_found:
            matched_num_delta = 1.0
        elif unaligned_found:
            unrelated_num_diff = 1.0

    # 2. Actor / Subject-Object Role Reversal check
    role_reversal = 0.0
    spo_consistent = 0.0
    if cand.subjects and cand.objects and mem.subjects and mem.objects:
        same_sub = bool(cand.subjects & mem.subjects)
        same_obj = bool(cand.objects & mem.objects)
        c_subj_in_m_obj = bool(cand.subjects & mem.objects)
        c_obj_in_m_subj = bool(cand.objects & mem.subjects)
        same_pred = bool(cand.predicates & mem.predicates)

        if (c_subj_in_m_obj and c_obj_in_m_subj) and not (same_sub and same_obj):
            role_reversal = 1.0
        elif same_sub and same_obj and same_pred:
            spo_consistent = 1.0
        elif (cand.subjects != mem.subjects or cand.objects != mem.objects) and same_pred and cos_sim >= 0.70:
            # e.g. same action ("announced" / "won") applied to different named actor/target
            if (cand.entities ^ mem.entities) and (cos_sim >= 0.70):
                role_reversal = 0.75

    # 3. Conditional Negation or Action-Antonym Polarity Flip
    cond_neg_flip = 0.0
    if compatible_context and (cand.negated != mem.negated):
        if bool(cand.predicates & mem.predicates) or lemma_overlap >= 0.40:
            cond_neg_flip = 1.0

    cond_action_flip = 0.0
    if compatible_context and cand.predicates and mem.predicates:
        # Require at least shared subject, object, entity, or >= 0.45 context similarity + >= 2 shared content lemmas
        actor_aligned = bool((cand.subjects & mem.subjects) or (cand.objects & mem.objects) or shared_ents or len(shared_lemmas) >= 2)
        if actor_aligned:
            for pos_set, neg_set in LEMMA_OPPOSITION_PAIRS:
                if (bool(cand.predicates & pos_set) and bool(mem.predicates & neg_set)) or (
                    bool(cand.predicates & neg_set) and bool(mem.predicates & pos_set)
                ):
                    cond_action_flip = 1.0
                    break

    # 4. Epistemic modality shift (hedged/conditional -> confirmed enactment)
    modality_shift = 0.0
    if compatible_context and mem.hedged_modality and (not cand.hedged_modality) and cand.confirmed_modality:
        modality_shift = 1.0

    return {
        "ent_jaccard": round(ent_jaccard, 4),
        "lemma_overlap": round(lemma_overlap, 4),
        "matched_num_delta": matched_num_delta,
        "unrelated_num_diff": unrelated_num_diff,
        "role_reversal": role_reversal,
        "spo_consistent": spo_consistent,
        "cond_neg_flip": cond_neg_flip,
        "cond_action_flip": cond_action_flip,
        "modality_shift": modality_shift,
    }


class RedesignedNoveltyEvaluator:
    """Conditional multi-signal novelty evaluator without hard overrides (Part 2)."""

    def __init__(self, ner: Optional[NamedEntityExtractor] = None):
        self.analyzer = PropositionAnalyzer(ner=ner)
        self.sfilter = SentenceFilter()

    def evaluate_sentence_against_memory(
        self,
        cand_text: str,
        cand_vec: np.ndarray,
        cand_prop: ExtractedProposition,
        mem_props: List[ExtractedProposition],
        mem_vecs: np.ndarray,
        redundancy_threshold: float = 0.75,
        novelty_threshold: float = 0.65,
        semantic_paraphrase_floor: float = 0.60,
    ) -> Dict[str, Any]:
        """Compute conditional proposition features and calibrated evidence status."""
        if cand_prop.is_boilerplate:
            return {
                "highest_sim": 0.0,
                "top3_mean_sim": 0.0,
                "base_novelty": 0.0,
                "informativeness": 0.0,
                "is_boilerplate": 1.0,
                "matched_num_delta": 0.0,
                "unrelated_num_diff": 0.0,
                "role_reversal": 0.0,
                "spo_consistent": 0.0,
                "cond_neg_flip": 0.0,
                "cond_action_flip": 0.0,
                "modality_shift": 0.0,
                "evidence_status": "BACKGROUND_INFORMATION",
                "conditional_score": 0.0,
            }

        info_score = self.sfilter.compute_informativeness_score(
            cand_text, is_headline=False, known_entities=list(cand_prop.entities)
        )
        if len(mem_props) == 0 or mem_vecs.shape[0] == 0:
            return {
                "highest_sim": 0.0,
                "top3_mean_sim": 0.0,
                "base_novelty": 1.0,
                "informativeness": info_score,
                "is_boilerplate": 0.0,
                "matched_num_delta": 0.0,
                "unrelated_num_diff": 0.0,
                "role_reversal": 0.0,
                "spo_consistent": 0.0,
                "cond_neg_flip": 0.0,
                "cond_action_flip": 0.0,
                "modality_shift": 0.0,
                "evidence_status": "POTENTIALLY_NEW",
                "conditional_score": round(float(info_score), 4),
            }

        sims = mem_vecs @ cand_vec
        top_k = min(3, len(sims))
        top_indices = np.argsort(-sims)[:top_k]
        best_idx = int(top_indices[0])
        highest_sim = float(np.clip(sims[best_idx], 0.0, 1.0))
        top3_mean = float(np.clip(np.mean(sims[top_indices]), 0.0, 1.0))
        base_novelty = max(0.0, min(1.0, 1.0 - highest_sim))

        # Compare against best semantic match (or top-2 best matches if slightly closer on SPO)
        pair_feats = compare_propositions(cand_prop, mem_props[best_idx], highest_sim)

        # Soft evidence fusion (NO hard 0.80/0.85 override on unrelated numbers!)
        delta_boost = (
            0.35 * pair_feats["matched_num_delta"]
            + 0.45 * pair_feats["role_reversal"]
            + 0.45 * pair_feats["cond_neg_flip"]
            + 0.40 * pair_feats["cond_action_flip"]
            + 0.25 * pair_feats["modality_shift"]
        )
        cond_score = min(1.0, max(0.0, base_novelty + delta_boost))

        # Evidence classification
        if delta_boost >= 0.30:
            status = "POTENTIALLY_NEW"
        elif pair_feats["spo_consistent"] > 0.5 and highest_sim >= semantic_paraphrase_floor and pair_feats["matched_num_delta"] == 0.0:
            status = "REPEATED_INFORMATION"
        elif highest_sim >= redundancy_threshold:
            status = "REPEATED_INFORMATION"
        elif highest_sim < novelty_threshold and info_score >= 0.30:
            status = "POTENTIALLY_NEW"
        else:
            status = "INSUFFICIENT_EVIDENCE"

        return {
            "highest_sim": round(highest_sim, 4),
            "top3_mean_sim": round(top3_mean, 4),
            "base_novelty": round(base_novelty, 4),
            "informativeness": round(float(info_score), 4),
            "is_boilerplate": 0.0,
            "matched_num_delta": pair_feats["matched_num_delta"],
            "unrelated_num_diff": pair_feats["unrelated_num_diff"],
            "role_reversal": pair_feats["role_reversal"],
            "spo_consistent": pair_feats["spo_consistent"],
            "cond_neg_flip": pair_feats["cond_neg_flip"],
            "cond_action_flip": pair_feats["cond_action_flip"],
            "modality_shift": pair_feats["modality_shift"],
            "evidence_status": status,
            "conditional_score": round(float(cond_score), 4),
        }
