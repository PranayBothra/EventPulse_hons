"""Entity text normalization, alias canonicalization, and honorific stripping."""

import re
import string
from typing import Dict, Optional, Set

# Common political, professional, and military titles/honorifics
HONORIFIC_PREFIXES = re.compile(
    r"^(?:President|Prime Minister|Minister|Senator|Congressman|Congresswoman|Governor|Mayor|Secretary|General|Admiral|Dr\.?|Mr\.?|Mrs\.?|Ms\.?|CEO|Founder|Director)\s+",
    re.IGNORECASE,
)

# Possessive suffixes ('s, ’s)
POSSESSIVE_SUFFIX = re.compile(r"['’]s$", re.IGNORECASE)

# Punctuation translation table (excluding hyphens inside words)
PUNCTUATION_STRIP = str.maketrans("", "", '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')

# Curated canonical alias mappings (case-insensitive keys -> canonical form)
CANONICAL_ALIASES: Dict[str, str] = {
    # Key political figures
    "trump": "donald trump",
    "donald trump": "donald trump",
    "biden": "joe biden",
    "joe biden": "joe biden",
    "albanese": "anthony albanese",
    "anthony albanese": "anthony albanese",
    "putin": "vladimir putin",
    "vladimir putin": "vladimir putin",
    "zelensky": "volodymyr zelensky",
    "zelenskiy": "volodymyr zelensky",
    "volodymyr zelensky": "volodymyr zelensky",
    # Key corporate & tech figures
    "jensen huang": "jensen huang",
    "huang": "jensen huang",
    "elon musk": "elon musk",
    "musk": "elon musk",
    "altman": "sam altman",
    "sam altman": "sam altman",
    # Organizations & Corporations
    "nvidia": "nvidia",
    "nvidia corp": "nvidia",
    "nvidia corporation": "nvidia",
    "spacex": "spacex",
    "space exploration technologies": "spacex",
    "oxfam": "oxfam",
    "unrwa": "unrwa",
    "united nations": "united nations",
    "un": "united nations",
    "european union": "european union",
    "eu": "european union",
    "megaspeed": "megaspeed",
    # Geopolitical entities / GPE
    "us": "united states",
    "u.s.": "united states",
    "usa": "united states",
    "united states": "united states",
    "united states of america": "united states",
    "america": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "britain": "united kingdom",
    "united kingdom": "united kingdom",
    "gaza": "gaza",
    "gaza strip": "gaza",
    "israel": "israel",
    "ukraine": "ukraine",
    "russia": "russia",
    "china": "china",
    "libya": "libya",
    "australia": "australia",
    # Recurring Global Events / Summits
    "cop31": "cop31",
    "cop 31": "cop31",
    "climate talks": "climate summit",
    "climate summit": "climate summit",
}

# Common noise words frequently misidentified as named entities in headlines
NON_ENTITY_WORDS: Set[str] = {
    "the", "a", "an", "this", "that", "these", "those",
    "what", "why", "how", "who", "when", "where",
    "breaking", "latest", "update", "live", "report", "news",
    "exclusive", "special", "analysis", "opinion",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "yesterday", "tomorrow", "tonight", "week", "month", "year",
    "first", "second", "third", "five", "ten", "one", "two",
}


def normalize_entity_name(raw_name: Optional[str]) -> str:
    """Normalize a recognized entity string into a clean, canonical form.

    Args:
        raw_name: Raw entity text string.

    Returns:
        Standardized, canonicalized entity identifier, or empty string if noise.
    """
    if not raw_name:
        return ""

    text = str(raw_name).strip()

    # Strip possessives: e.g. "Huang's" -> "Huang"
    text = POSSESSIVE_SUFFIX.sub("", text)

    # Strip honorific prefixes: e.g. "President Trump" -> "Trump"
    text = HONORIFIC_PREFIXES.sub("", text)

    # Remove outer punctuation and clean characters
    text = text.translate(PUNCTUATION_STRIP).strip()

    # Standardize casing and whitespace
    text_lower = " ".join(text.lower().split())

    if not text_lower or len(text_lower) < 2:
        return ""

    if text_lower in NON_ENTITY_WORDS:
        return ""

    # Check canonical dictionary
    if text_lower in CANONICAL_ALIASES:
        return CANONICAL_ALIASES[text_lower]

    return text_lower
