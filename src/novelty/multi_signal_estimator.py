"""Multi-signal sentence novelty estimation for EventPulse.

Integrates:
1. Dense bi-encoder semantic similarity (all-MiniLM-L6-v2)
2. Numerical and metric change detection (quantities, casualties, percentages, currency)
3. Action and outcome changes (ruling verbs, decisions, policy shifts)
4. Polarity and negation mismatch (approved/rejected, guilty/not guilty)
5. Enhanced information-bearing and promotional boilerplate filtering
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# Action and outcome opposition pairs
OPPOSITION_PAIRS: List[Tuple[Set[str], Set[str]]] = [
    ({"approved", "approve", "approves", "approving"}, {"rejected", "reject", "rejects", "rejecting"}),
    ({"passed", "pass", "passes", "passing"}, {"failed", "fail", "fails", "failing"}),
    ({"guilty"}, {"not guilty", "innocent", "acquitted", "acquit"}),
    ({"convicted", "convict", "convicts"}, {"acquitted", "acquit", "acquits"}),
    ({"won", "win", "wins", "winning"}, {"lost", "lose", "loses", "losing"}),
    ({"increased", "increase", "increases", "rose", "surged"}, {"decreased", "decrease", "decreases", "fell", "dropped", "plummeted"}),
    ({"upheld", "uphold", "upholds"}, {"overturned", "overturn", "overturns", "reversed", "reverse"}),
    ({"confirmed", "confirm", "confirms"}, {"denied", "deny", "denies"}),
    ({"agreed", "agree", "agrees"}, {"disagreed", "disagree", "disagrees"}),
    ({"accepted", "accept", "accepts"}, {"declined", "decline", "declines", "refused"}),
    ({"resumed", "resume", "resumes", "restarted"}, {"collapsed", "collapse", "collapses", "halted", "suspended", "stalled"}),
]

NEGATION_TOKENS: Set[str] = {
    "not", "never", "no", "neither", "cannot", "didn't", "wasn't", "weren't",
    "hasn't", "haven't", "isn't", "aren't", "barely", "hardly", "failed to"
}

NUMBER_PATTERN = re.compile(
    r"\b(?:\$|€|£)?\d+(?:[.,]\d+)*(?:\s*(?:%|percent|billion|million|thousand|hundred|deaths|casualties|injured|dead|people|votes|points|seats|miles|km|meters|tonnes|dollars|euros|cents|pence))?\b",
    re.IGNORECASE,
)

NUMBER_WORDS_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
    "thousand": "1000", "million": "1000000", "billion": "1000000000",
}

PROMOTIONAL_PATTERNS = [
    re.compile(r"\bdownload the (?:channel \d+|\w+) (?:news )?(?:app|application)\b", re.IGNORECASE),
    re.compile(r"\b(listen on (?:spotify|apple|google)|watch live on|stream now)\b", re.IGNORECASE),
    re.compile(r"\b(follow us on (?:twitter|facebook|instagram|tiktok))\b", re.IGNORECASE),
    re.compile(r"\b(sign up for|subscribe to|read the full (?:story|article))\b", re.IGNORECASE),
    re.compile(r"\b(all rights reserved|copyright \d{4}|click here to)\b", re.IGNORECASE),
]


@dataclass
class MultiSignalNoveltyResult:
    """Rich, explainable output of multi-signal novelty assessment."""
    candidate_sentence: str
    highest_similarity: float
    most_similar_text: Optional[str]
    numerical_change_detected: bool
    numerical_deltas: Tuple[List[str], List[str]]
    polarity_flip_detected: bool
    polarity_reason: str
    is_promotional_or_boilerplate: bool
    composite_novelty_score: float
    evidence_status: str  # POTENTIALLY_NEW, REPEATED_INFORMATION, BACKGROUND_INFORMATION, INSUFFICIENT_EVIDENCE
    confidence: float
    explanation: str


class MultiSignalNoveltyEstimator:
    """Estimates information novelty by combining dense semantics, entity attributes, numbers, and polarity."""

    def __init__(
        self,
        redundancy_threshold: float = 0.75,
        novelty_threshold: float = 0.65,
        top_k_similarity: int = 1,
    ):
        self.redundancy_threshold = redundancy_threshold
        self.novelty_threshold = novelty_threshold
        self.top_k_similarity = top_k_similarity

    def normalize_number_words(self, text: str) -> str:
        """Replace spelled-out number words with digits for robust extraction."""
        tokens = text.split()
        normalized = []
        for t in tokens:
            clean_t = re.sub(r"[^\w]", "", t.lower())
            if clean_t in NUMBER_WORDS_MAP:
                normalized.append(NUMBER_WORDS_MAP[clean_t])
            else:
                normalized.append(t)
        return " ".join(normalized)

    def extract_numbers_and_quantities(self, text: str) -> List[str]:
        """Extract all numerical, financial, and metric quantities from text."""
        norm_text = self.normalize_number_words(text)
        matches = NUMBER_PATTERN.findall(norm_text)
        cleaned = []
        for m in matches:
            c = m.strip().lower()
            if c:
                # Keep core digit representation
                digits = re.findall(r"\d+", c)
                if digits:
                    cleaned.append(digits[0])
                else:
                    cleaned.append(c)
        return cleaned

    def detect_numerical_difference(
        self, cand_text: str, hist_text: str
    ) -> Tuple[bool, List[str], List[str]]:
        """Detect whether critical numerical quantities differ between proximate sentences."""
        cand_nums = self.extract_numbers_and_quantities(cand_text)
        hist_nums = self.extract_numbers_and_quantities(hist_text)

        if not cand_nums and not hist_nums:
            return False, [], []

        cand_set = set(cand_nums)
        hist_set = set(hist_nums)
        diff_cand = [n for n in cand_nums if n not in hist_set]
        diff_hist = [n for n in hist_nums if n not in cand_set]

        # A numerical change requires at least one different number on each side
        if diff_cand and diff_hist:
            return True, diff_cand, diff_hist
        return False, diff_cand, diff_hist

    def detect_polarity_and_action_flip(
        self, cand_text: str, hist_text: str
    ) -> Tuple[bool, str]:
        """Detect action flips (approved/rejected) or negation flips (guilty/not guilty)."""
        c_words = set(re.findall(r"\b\w+\b", cand_text.lower()))
        h_words = set(re.findall(r"\b\w+\b", hist_text.lower()))

        # 1. Action opposition pairs
        for pos_set, neg_set in OPPOSITION_PAIRS:
            c_pos = bool(c_words & pos_set)
            c_neg = bool(c_words & neg_set)
            h_pos = bool(h_words & pos_set)
            h_neg = bool(h_words & neg_set)

            if (c_pos and h_neg) or (c_neg and h_pos):
                return True, "OPPOSITION_ACTION_FLIP"

        # 2. Negation token mismatch on shared content
        c_has_neg = bool(c_words & NEGATION_TOKENS) or "not guilty" in cand_text.lower()
        h_has_neg = bool(h_words & NEGATION_TOKENS) or "not guilty" in hist_text.lower()

        if c_has_neg != h_has_neg:
            # Check lexical overlap of substantive words (>= 40% shared content)
            c_sub = {w for w in c_words if len(w) > 3 and w not in NEGATION_TOKENS}
            h_sub = {w for w in h_words if len(w) > 3 and w not in NEGATION_TOKENS}
            if c_sub and h_sub:
                overlap = len(c_sub & h_sub) / min(len(c_sub), len(h_sub))
                if overlap >= 0.40:
                    return True, f"NEGATION_POLARITY_FLIP (overlap={overlap:.2f})"

        return False, "NONE"

    def is_promotional_or_boilerplate(self, text: str) -> bool:
        """Check if candidate sentence is promotional boilerplate or attribution noise."""
        for pat in PROMOTIONAL_PATTERNS:
            if pat.search(text):
                return True
        return False

    def estimate_novelty(
        self,
        candidate_text: str,
        candidate_embedding: np.ndarray,
        memory_texts: List[str],
        memory_embeddings: np.ndarray,
        informativeness_score: float = 0.50,
    ) -> MultiSignalNoveltyResult:
        """Compute multi-signal novelty score with evidence classification."""
        # 1. Boilerplate check
        if self.is_promotional_or_boilerplate(candidate_text):
            return MultiSignalNoveltyResult(
                candidate_sentence=candidate_text,
                highest_similarity=0.0,
                most_similar_text=None,
                numerical_change_detected=False,
                numerical_deltas=([], []),
                polarity_flip_detected=False,
                polarity_reason="BOILERPLATE",
                is_promotional_or_boilerplate=True,
                composite_novelty_score=0.0,
                evidence_status="BACKGROUND_INFORMATION",
                confidence=0.90,
                explanation="Filtered as publisher promotional or boilerplate text.",
            )

        # 2. Memory check
        if len(memory_texts) == 0 or memory_embeddings.shape[0] == 0:
            return MultiSignalNoveltyResult(
                candidate_sentence=candidate_text,
                highest_similarity=0.0,
                most_similar_text=None,
                numerical_change_detected=False,
                numerical_deltas=([], []),
                polarity_flip_detected=False,
                polarity_reason="INITIAL_MEMORY",
                is_promotional_or_boilerplate=False,
                composite_novelty_score=round(informativeness_score, 4),
                evidence_status="POTENTIALLY_NEW",
                confidence=1.0,
                explanation="First recorded information in event memory.",
            )

        # Ensure unit normalization
        cand_norm = np.linalg.norm(candidate_embedding)
        cand_vec = candidate_embedding / cand_norm if cand_norm > 0 else candidate_embedding

        # Dense similarity against memory
        sims = np.dot(memory_embeddings, cand_vec)
        best_idx = int(np.argmax(sims))
        highest_sim = float(sims[best_idx])
        most_similar_text = memory_texts[best_idx]

        # 3. Numerical difference check against most similar sentence
        num_diff, c_nums, h_nums = self.detect_numerical_difference(candidate_text, most_similar_text)

        # 4. Polarity and action flip check against most similar sentence
        pol_diff, pol_reason = self.detect_polarity_and_action_flip(candidate_text, most_similar_text)

        # 5. Composite Novelty Calculation
        # Base novelty from semantic distance
        base_novelty = max(0.0, min(1.0, 1.0 - highest_sim))

        # Adjust for critical factual updates
        confidence = 0.80
        reasons = []

        if pol_diff:
            # Overrides high cosine similarity: polarity has flipped!
            adjusted_novelty = max(0.85, base_novelty)
            evidence_status = "POTENTIALLY_NEW"
            confidence = 0.95
            reasons.append(f"Action/polarity reversal detected ({pol_reason})")
        elif num_diff:
            # Overrides high cosine similarity: numerical update occurred!
            adjusted_novelty = max(0.80, base_novelty)
            evidence_status = "POTENTIALLY_NEW"
            confidence = 0.90
            reasons.append(f"Numerical update detected ({c_nums} vs {h_nums})")
        else:
            # Standard thresholding
            adjusted_novelty = base_novelty * informativeness_score
            if highest_sim >= self.redundancy_threshold:
                evidence_status = "REPEATED_INFORMATION"
                reasons.append(f"High semantic overlap ({highest_sim:.2f} >= {self.redundancy_threshold})")
            elif highest_sim < self.novelty_threshold and informativeness_score >= 0.35:
                evidence_status = "POTENTIALLY_NEW"
                reasons.append(f"Distinct proposition ({highest_sim:.2f} < {self.novelty_threshold})")
            else:
                evidence_status = "INSUFFICIENT_EVIDENCE"
                reasons.append(f"Ambiguous similarity ({highest_sim:.2f})")

        clamped_score = max(0.0, min(1.0, round(float(adjusted_novelty), 4)))
        explanation = "; ".join(reasons)

        return MultiSignalNoveltyResult(
            candidate_sentence=candidate_text,
            highest_similarity=round(highest_sim, 4),
            most_similar_text=most_similar_text,
            numerical_change_detected=num_diff,
            numerical_deltas=(c_nums, h_nums),
            polarity_flip_detected=pol_diff,
            polarity_reason=pol_reason,
            is_promotional_or_boilerplate=False,
            composite_novelty_score=clamped_score,
            evidence_status=evidence_status,
            confidence=confidence,
            explanation=explanation,
        )
