"""Ablation configurations and experiment runner for EventPulse."""

from dataclasses import dataclass
from typing import Dict
from src.recommendation.ranking import RecommendationWeights


@dataclass
class AblationConfig:
    """Specification of an ablation experiment variant."""
    code: str
    name: str
    description: str
    weights: RecommendationWeights
    use_dual_profile: bool = True
    apply_mmr: bool = True


ABLATION_VARIANTS: Dict[str, AblationConfig] = {
    "A_FULL": AblationConfig(
        code="A_FULL",
        name="Variant A: Full EventPulse",
        description="Full multi-objective ranking with dual profiles, event novelty, user novelty, exposure penalty, freshness, and MMR reranking.",
        weights=RecommendationWeights(
            w_long=0.20,
            w_short=0.25,
            w_event_novelty=0.20,
            w_user_novelty=0.15,
            w_freshness=0.15,
            w_source_diversity=0.05,
            w_exposure_penalty=0.20,
            w_redundancy_penalty=0.15,
            mmr_lambda=0.70,
        ),
        use_dual_profile=True,
        apply_mmr=True,
    ),
    "B_NO_EVENT_NOVELTY": AblationConfig(
        code="B_NO_EVENT_NOVELTY",
        name="Variant B: Without Event Novelty",
        description="Disables incremental event novelty (w_event_novelty = 0.0).",
        weights=RecommendationWeights(
            w_long=0.25,
            w_short=0.30,
            w_event_novelty=0.0,
            w_user_novelty=0.20,
            w_freshness=0.20,
            w_source_diversity=0.05,
            w_exposure_penalty=0.20,
            w_redundancy_penalty=0.15,
            mmr_lambda=0.70,
        ),
        use_dual_profile=True,
        apply_mmr=True,
    ),
    "C_NO_USER_NOVELTY": AblationConfig(
        code="C_NO_USER_NOVELTY",
        name="Variant C: Without User Novelty",
        description="Disables personalized user novelty (w_user_novelty = 0.0).",
        weights=RecommendationWeights(
            w_long=0.25,
            w_short=0.30,
            w_event_novelty=0.25,
            w_user_novelty=0.0,
            w_freshness=0.15,
            w_source_diversity=0.05,
            w_exposure_penalty=0.20,
            w_redundancy_penalty=0.15,
            mmr_lambda=0.70,
        ),
        use_dual_profile=True,
        apply_mmr=True,
    ),
    "D_STATIC_PROFILE": AblationConfig(
        code="D_STATIC_PROFILE",
        name="Variant D: Without Dual Profile",
        description="Disables exponential short-term profile decay; uses single static history centroid.",
        weights=RecommendationWeights(
            w_long=0.45,
            w_short=0.0,
            w_event_novelty=0.20,
            w_user_novelty=0.15,
            w_freshness=0.15,
            w_source_diversity=0.05,
            w_exposure_penalty=0.20,
            w_redundancy_penalty=0.15,
            mmr_lambda=0.70,
        ),
        use_dual_profile=False,
        apply_mmr=True,
    ),
    "E_NO_FRESHNESS": AblationConfig(
        code="E_NO_FRESHNESS",
        name="Variant E: Without Freshness Decay",
        description="Disables temporal freshness decay (w_freshness = 0.0).",
        weights=RecommendationWeights(
            w_long=0.25,
            w_short=0.30,
            w_event_novelty=0.25,
            w_user_novelty=0.15,
            w_freshness=0.0,
            w_source_diversity=0.05,
            w_exposure_penalty=0.20,
            w_redundancy_penalty=0.15,
            mmr_lambda=0.70,
        ),
        use_dual_profile=True,
        apply_mmr=True,
    ),
    "F_NO_PENALTIES": AblationConfig(
        code="F_NO_PENALTIES",
        name="Variant F: Without Exposure & Redundancy Penalties",
        description="Disables exposure fatigue and redundancy penalty subtraction (w_exposure = 0.0, w_redundancy = 0.0).",
        weights=RecommendationWeights(
            w_long=0.25,
            w_short=0.30,
            w_event_novelty=0.20,
            w_user_novelty=0.15,
            w_freshness=0.10,
            w_source_diversity=0.0,
            w_exposure_penalty=0.0,
            w_redundancy_penalty=0.0,
            mmr_lambda=0.70,
        ),
        use_dual_profile=True,
        apply_mmr=True,
    ),
    "G_NO_MMR": AblationConfig(
        code="G_NO_MMR",
        name="Variant G: Without MMR Reranking",
        description="Disables maximal marginal relevance diversity reranking (mmr_lambda = 1.0).",
        weights=RecommendationWeights(
            w_long=0.20,
            w_short=0.25,
            w_event_novelty=0.20,
            w_user_novelty=0.15,
            w_freshness=0.15,
            w_source_diversity=0.05,
            w_exposure_penalty=0.20,
            w_redundancy_penalty=0.15,
            mmr_lambda=1.0,
        ),
        use_dual_profile=True,
        apply_mmr=False,
    ),
}
