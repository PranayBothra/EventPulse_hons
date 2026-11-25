"""Offline empirical benchmark retrieval endpoints."""

import json
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from src.config import get_project_root

router = APIRouter(prefix="/api/benchmarks", tags=["Offline Benchmarks"])


def _load_report_json(filename: str) -> Dict[str, Any]:
    path = get_project_root() / "reports" / filename
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Stored offline benchmark report '{filename}' not found.",
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.get("/mind")
def get_mind_benchmark() -> Dict[str, Any]:
    data = _load_report_json("mind_benchmark_results.json")
    return {
        "evaluation_scope": "offline_benchmark",
        "source_file": "reports/mind_benchmark_results.json",
        "note": "Pre-computed offline MIND evaluation metrics; distinct from live news stream metrics.",
        "results": data,
    }


@router.get("/corpus-novelty")
def get_corpus_novelty_benchmark() -> Dict[str, Any]:
    data = _load_report_json("corpus_novelty_results.json")
    return {
        "evaluation_scope": "offline_benchmark",
        "source_file": "reports/corpus_novelty_results.json",
        "note": "Pre-computed offline corpus novelty saturation metrics; distinct from live interactive usage.",
        "results": data,
    }
