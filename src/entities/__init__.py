"""Named entity extraction and normalization for EventPulse."""

from src.entities.extractor import NamedEntityExtractor
from src.entities.normalizer import normalize_entity_name

__all__ = ["NamedEntityExtractor", "normalize_entity_name"]
