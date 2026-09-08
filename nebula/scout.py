"""Base-Model Scout: rank Hugging Face base-model candidates for a Centinela model.

Pure ranking logic — callers (scripts/00_scout.py) fetch raw model records from
the HF Hub API and map them to Candidate before calling rank_candidates. No network
here so the ranking stays deterministic and unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Constraints:
    max_params_b: float
    allowed_licenses: list[str]
    language: str
    pipeline_tag: str = "text-generation"


@dataclass
class Candidate:
    model_id: str
    license: str | None
    params_b: float | None
    languages: list[str]
    pipeline_tag: str | None
    downloads: int
    likes: int
    leaderboard_score: float | None = None


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float
    rationale: str


def _passes(c: Candidate, k: Constraints) -> bool:
    if c.license not in k.allowed_licenses:
        return False
    if c.params_b is not None and c.params_b > k.max_params_b:
        return False
    if c.pipeline_tag is not None and c.pipeline_tag != k.pipeline_tag:
        return False
    if c.languages and k.language not in c.languages:
        return False
    return True


def _score(c: Candidate) -> float:
    leaderboard = c.leaderboard_score if c.leaderboard_score is not None else 0.0
    popularity = math.log10(c.downloads + 1) + math.log10(c.likes + 1)
    return 1.5 * leaderboard + 0.5 * popularity


def _rationale(c: Candidate, k: Constraints) -> str:
    size = f"{c.params_b}B" if c.params_b is not None else "size unknown"
    return (
        f"license={c.license}, {size} <= {k.max_params_b}B budget, "
        f"lang '{k.language}' ok, downloads={c.downloads}, likes={c.likes}"
    )


def rank_candidates(
    candidates: list[Candidate], constraints: Constraints
) -> list[ScoredCandidate]:
    """Filter by hard constraints, then sort survivors by descending score."""
    survivors = [c for c in candidates if _passes(c, constraints)]
    scored = [
        ScoredCandidate(candidate=c, score=_score(c), rationale=_rationale(c, constraints))
        for c in survivors
    ]
    scored.sort(key=lambda s: (-s.score, s.candidate.model_id))
    return scored
