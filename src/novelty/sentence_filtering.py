"""Sentence filtering, boilerplate detection, and informativeness scoring for EventPulse."""

import re
from typing import List, Optional, Tuple


class SentenceFilter:
    """Filters noisy boilerplate sentences and scores factual informativeness."""

    # Common journalistic and web boilerplate patterns
    BOILERPLATE_PATTERNS = [
        re.compile(r"^(photo|image|picture|video|source|credit|courtesy|caption)\s*:", re.IGNORECASE),
        re.compile(r"\b(photo by|image copyright|getty images|ap photo|afp via|reuters\/)\b", re.IGNORECASE),
        re.compile(r"\b(sign up for|subscribe to|read the full (story|article)|follow us on)\b", re.IGNORECASE),
        re.compile(r"\b(all rights reserved|copyright \d{4}|click here to|download our app)\b", re.IGNORECASE),
        re.compile(r"^(published|updated|last modified|posted)\s*:\s*\w+", re.IGNORECASE),
        re.compile(r"^(by\s+[a-z\s]+|editor's note:|corrections?:)", re.IGNORECASE),
        re.compile(r"\b(terms of service|privacy policy|cookie settings|advertisement)\b", re.IGNORECASE),
        re.compile(r"^(share this article|listen to this story|related stories|more on this:)", re.IGNORECASE),
    ]

    # Temporal patterns
    TEMPORAL_PATTERNS = re.compile(
        r"\b(yesterday|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|october|november|december|"
        r"morning|afternoon|evening|night|\d{4}|\d{1,2}:\d{2})\b",
        re.IGNORECASE,
    )

    # Quantitative / numerical patterns
    NUMERICAL_PATTERNS = re.compile(r"(\$|€|£|¥|\b\d+(\.\d+)?%?|\b(million|billion|trillion|hundred|thousand)\b)", re.IGNORECASE)

    # Reporting / action verbs common in substantive event developments
    ACTION_VERBS = re.compile(
        r"\b(announced|signed|confirmed|denied|declared|launched|agreed|voted|approved|"
        r"rejected|reported|killed|injured|arrested|struck|collapsed|investigating|"
        r"increased|decreased|surged|plummeted|unveiled|issued|warned|demanded)\b",
        re.IGNORECASE,
    )

    # Vague rhetorical fluff to discount
    RHETORICAL_FLUFF = re.compile(
        r"\b(needless to say|it goes without saying|time will tell|in other news|"
        r"as you may know|stay tuned|only time will tell|as previously reported)\b",
        re.IGNORECASE,
    )

    def __init__(self, min_char_length: int = 15, min_word_count: int = 4, max_word_count: int = 120):
        self.min_char_length = min_char_length
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count

    def clean_sentence(self, text: str) -> str:
        """Clean whitespace and unwanted bounding quotation marks."""
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        # Remove surrounding quotes
        cleaned = re.sub(r"^[\"\'“‘]+|[\"\'”’]+$", "", cleaned).strip()
        return cleaned

    def is_valid_sentence(self, text: str) -> Tuple[bool, str]:
        """Check if a candidate sentence is information-bearing and not noise.

        Returns (is_valid, reason).
        """
        cleaned = self.clean_sentence(text)
        if not cleaned:
            return False, "EMPTY"

        if len(cleaned) < self.min_char_length:
            return False, f"TOO_SHORT_CHARS ({len(cleaned)} < {self.min_char_length})"

        words = cleaned.split()
        if len(words) < self.min_word_count:
            return False, f"TOO_FEW_WORDS ({len(words)} < {self.min_word_count})"

        if len(words) > self.max_word_count:
            return False, f"TOO_MANY_WORDS ({len(words)} > {self.max_word_count})"

        # Check boilerplate patterns
        for pattern in self.BOILERPLATE_PATTERNS:
            if pattern.search(cleaned):
                return False, f"BOILERPLATE_MATCH: {pattern.pattern}"

        # Filter out sentences that are mostly non-alphabetic
        alpha_count = sum(c.isalpha() for c in cleaned)
        if alpha_count / max(len(cleaned), 1) < 0.5:
            return False, "LOW_ALPHABETIC_RATIO"

        return True, "VALID"

    def compute_informativeness_score(
        self,
        text: str,
        is_headline: bool = False,
        known_entities: Optional[List[str]] = None,
    ) -> float:
        """Compute an informativeness score in [0.05, 1.0] based on factual cues.

        Heuristic grounding:
        - Headlines provide dense topic summaries: +0.20
        - Quantitative figures / metrics: +0.20
        - Named entities / proper nouns: +0.25
        - Action / reporting verbs: +0.15
        - Temporal anchors: +0.10
        - Ideal sentence length (10 to 40 words): +0.10
        - Penalties for rhetorical fluff: -0.30
        """
        cleaned = self.clean_sentence(text)
        if not cleaned:
            return 0.05

        score = 0.20  # Base prior for valid sentences

        if is_headline:
            score += 0.20

        # Quantitative metrics
        if self.NUMERICAL_PATTERNS.search(cleaned):
            score += 0.20

        # Temporal anchors
        if self.TEMPORAL_PATTERNS.search(cleaned):
            score += 0.10

        # Action / reporting verbs
        if self.ACTION_VERBS.search(cleaned):
            score += 0.15

        # Named entities / Capitalized tokens
        words = cleaned.split()
        proper_nouns = [w for w in words[1:] if w and w[0].isupper() and w.isalpha()]
        if proper_nouns or (known_entities and len(known_entities) > 0):
            score += 0.25

        # Sentence length quality
        num_words = len(words)
        if 8 <= num_words <= 45:
            score += 0.10
        elif num_words < 6 or num_words > 60:
            score -= 0.10

        # Fluff penalty
        if self.RHETORICAL_FLUFF.search(cleaned):
            score -= 0.30

        return max(0.05, min(1.0, round(score, 4)))
