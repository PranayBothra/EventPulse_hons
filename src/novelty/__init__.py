"""Information novelty detection, sentence processing, and memory modules for EventPulse."""

from src.novelty.sentence_filtering import SentenceFilter
from src.novelty.sentence_extraction import ExtractedSentence, SentenceExtractor, SentencePipeline
from src.novelty.event_memory import MemorySentence, EventMemoryManager
from src.novelty.novelty_scoring import SentenceNoveltyResult, SentenceNoveltyScorer
from src.novelty.novelty_aggregation import ArticleNoveltySummary, EventNoveltyEngine
from src.novelty.multi_signal_estimator import MultiSignalNoveltyResult, MultiSignalNoveltyEstimator

__all__ = [
    "SentenceFilter",
    "ExtractedSentence",
    "SentenceExtractor",
    "SentencePipeline",
    "MemorySentence",
    "EventMemoryManager",
    "SentenceNoveltyResult",
    "SentenceNoveltyScorer",
    "ArticleNoveltySummary",
    "EventNoveltyEngine",
    "MultiSignalNoveltyResult",
    "MultiSignalNoveltyEstimator",
]
