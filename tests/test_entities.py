"""Unit tests for entity extraction and normalisation."""

import pytest
from src.entities.extractor import NamedEntityExtractor
from src.entities.normalizer import normalize_entity_name


class TestEntityNormalizer:
    def test_strip_honorifics(self):
        assert normalize_entity_name("President Trump") == "donald trump"
        assert normalize_entity_name("CEO Jensen Huang") == "jensen huang"
        assert normalize_entity_name("Prime Minister Albanese") == "anthony albanese"
        assert normalize_entity_name("Dr. Smith") == "smith"

    def test_strip_possessives(self):
        assert normalize_entity_name("Huang's") == "jensen huang"
        assert normalize_entity_name("Nvidia’s") == "nvidia"
        assert normalize_entity_name("SpaceX's") == "spacex"

    def test_canonical_aliases(self):
        assert normalize_entity_name("US") == "united states"
        assert normalize_entity_name("U.S.A.") == "united states"
        assert normalize_entity_name("America") == "united states"
        assert normalize_entity_name("UK") == "united kingdom"
        assert normalize_entity_name("EU") == "european union"
        assert normalize_entity_name("Gaza Strip") == "gaza"
        assert normalize_entity_name("Cop31") == "cop31"

    def test_noise_filtering(self):
        assert normalize_entity_name("The") == ""
        assert normalize_entity_name("Breaking") == ""
        assert normalize_entity_name("Monday") == ""
        assert normalize_entity_name("A") == ""
        assert normalize_entity_name(None) == ""


class TestNamedEntityExtractor:
    def test_extract_spacy_if_available(self):
        extractor = NamedEntityExtractor(use_spacy=True)
        text = "Donald Trump welcomed Jensen Huang to discuss Nvidia in Washington."
        categorized = extractor.extract(text)

        assert "donald trump" in categorized["PERSON"]
        assert "jensen huang" in categorized["PERSON"]
        assert "nvidia" in categorized["ORG"]
        assert "washington" in categorized["GPE"]

        flat = extractor.extract_flat(text)
        assert "donald trump" in flat
        assert "nvidia" in flat

    def test_extract_heuristic_fallback(self):
        # Explicitly test heuristic fallback mode
        extractor = NamedEntityExtractor(use_spacy=False)
        assert extractor.nlp is None

        hl = "SpaceX faces lawsuit over Grok AI tool at COP31 climate summit"
        categorized = extractor.extract(hl)

        assert "spacex" in categorized["ORG"]
        assert "cop31" in categorized["EVENT"]
        assert "grok" in categorized["PRODUCT"]

    def test_extract_empty_or_whitespace(self):
        extractor = NamedEntityExtractor(use_spacy=False)
        categorized = extractor.extract("", "   ")
        for cat, entities in categorized.items():
            assert len(entities) == 0

        flat = extractor.extract_flat(None)
        assert len(flat) == 0
