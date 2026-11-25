"""Named entity extractor supporting spaCy with an intelligent heuristic fallback."""

import logging
import re
from typing import Dict, List, Optional, Set

from src.entities.normalizer import (
    CANONICAL_ALIASES,
    NON_ENTITY_WORDS,
    normalize_entity_name,
)

logger = logging.getLogger("eventpulse.entities")

# Regular expression pattern to capture capitalized multi-word phrases and acronyms
CAPITALIZED_PHRASE_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9]*(?:[-'\s]+[A-Z][a-zA-Z0-9]*)*\b"
)
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}[0-9]*\b")


class NamedEntityExtractor:
    """Robust Named Entity Extractor using spaCy with fallback heuristics."""

    _shared_nlp = None

    def __init__(self, use_spacy: bool = True):
        self.nlp = None
        self.use_spacy = use_spacy

        if use_spacy:
            if NamedEntityExtractor._shared_nlp is None:
                try:
                    import spacy
                    NamedEntityExtractor._shared_nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model 'en_core_web_sm' for NER.")
                except Exception as e:
                    logger.warning(
                        "spaCy model 'en_core_web_sm' could not be loaded (%s). Using heuristic fallback.",
                        e,
                    )
            self.nlp = NamedEntityExtractor._shared_nlp

    def extract(
        self, headline: str, description: Optional[str] = None
    ) -> Dict[str, Set[str]]:
        """Extract categorized named entities from headline and description.

        Args:
            headline: Article headline text.
            description: Optional summary or body snippet.

        Returns:
            Dictionary mapping category ('PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT', 'OTHER')
            to a set of normalized canonical entity strings.
        """
        combined_text = f"{headline or ''}. {description or ''}".strip()
        if not combined_text or combined_text == ".":
            return {
                "PERSON": set(),
                "ORG": set(),
                "GPE": set(),
                "EVENT": set(),
                "PRODUCT": set(),
                "OTHER": set(),
            }

        if self.nlp is not None:
            return self._extract_spacy(combined_text)
        else:
            return self._extract_heuristic(combined_text)

    def extract_flat(
        self, headline: str, description: Optional[str] = None
    ) -> Set[str]:
        """Extract all unique normalized entities into a single flat set.

        Useful for fast Jaccard overlap comparisons between articles.
        """
        categorized = self.extract(headline, description)
        flat = set()
        for entities in categorized.values():
            flat.update(entities)
        return flat

    def _extract_spacy(self, text: str) -> Dict[str, Set[str]]:
        """Extract and categorize entities using spaCy NER."""
        doc = self.nlp(text)
        categories: Dict[str, Set[str]] = {
            "PERSON": set(),
            "ORG": set(),
            "GPE": set(),
            "EVENT": set(),
            "PRODUCT": set(),
            "OTHER": set(),
        }

        category_map = {
            "PERSON": "PERSON",
            "ORG": "ORG",
            "GPE": "GPE",
            "LOC": "GPE",
            "EVENT": "EVENT",
            "PRODUCT": "PRODUCT",
            "WORK_OF_ART": "PRODUCT",
            "FAC": "GPE",
            "LAW": "OTHER",
            "NORP": "OTHER",
        }

        for ent in doc.ents:
            normalized = normalize_entity_name(ent.text)
            if not normalized:
                continue

            target_cat = category_map.get(ent.label_, "OTHER")
            categories[target_cat].add(normalized)

        # Also capture known gazetteer terms that spaCy might miss in headlines (e.g. COP31, Grok)
        self._supplement_from_gazetteer(text, categories)
        return categories

    def _extract_heuristic(self, text: str) -> Dict[str, Set[str]]:
        """Fallback extraction using regex pattern matching and gazetteer lookup."""
        categories: Dict[str, Set[str]] = {
            "PERSON": set(),
            "ORG": set(),
            "GPE": set(),
            "EVENT": set(),
            "PRODUCT": set(),
            "OTHER": set(),
        }

        # 1. Gazetteer exact matches
        self._supplement_from_gazetteer(text, categories)

        # 2. Capitalized phrases
        matches = CAPITALIZED_PHRASE_RE.findall(text) + ACRONYM_RE.findall(text)
        for match in matches:
            normalized = normalize_entity_name(match)
            if not normalized or normalized in NON_ENTITY_WORDS:
                continue

            # Assign to category if not already placed
            placed = any(normalized in cat_set for cat_set in categories.values())
            if not placed:
                categories["OTHER"].add(normalized)

        return categories

    def _supplement_from_gazetteer(
        self, text: str, categories: Dict[str, Set[str]]
    ) -> None:
        """Scan text for high-value known domain entities."""
        lower_text = text.lower()
        # Events
        if "cop31" in lower_text or "cop 31" in lower_text:
            categories["EVENT"].add("cop31")
        if "climate summit" in lower_text or "climate talks" in lower_text:
            categories["EVENT"].add("climate summit")

        # Key Org / People
        if "nvidia" in lower_text:
            categories["ORG"].add("nvidia")
        if "spacex" in lower_text:
            categories["ORG"].add("spacex")
        if "oxfam" in lower_text:
            categories["ORG"].add("oxfam")
        if "unrwa" in lower_text:
            categories["ORG"].add("unrwa")
        if "jensen huang" in lower_text or "huang" in lower_text:
            categories["PERSON"].add("jensen huang")
        if "donald trump" in lower_text or "trump" in lower_text:
            categories["PERSON"].add("donald trump")
        if "musk" in lower_text:
            categories["PERSON"].add("elon musk")
        if "grok" in lower_text:
            categories["PRODUCT"].add("grok")
        if "starshield" in lower_text:
            categories["PRODUCT"].add("starshield")
