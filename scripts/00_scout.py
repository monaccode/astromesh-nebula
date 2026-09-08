"""Base-Model Scout CLI: query the HF Hub for candidate base models and rank them.

    uv run python scripts/00_scout.py --task text-classification --vertical finanzas \
        --language es --max-params-b 8 --top 5

Prints a ranked shortlist with rationale. The human picks one to set as base_model
in configs/train-<name>.yaml.
"""

import argparse
import re

from huggingface_hub import HfApi

from nebula.scout import Candidate, Constraints, rank_candidates

# Best-effort: matches "-4B"/"-8b"; may misread ids like "bloom-7b1" (reads 7, not 7.1).
_SIZE_RE = re.compile(r"[-_/](\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def _params_b_from_id(model_id: str) -> float | None:
    match = _SIZE_RE.search(model_id)
    return float(match.group(1)) if match else None


def _license_from(info) -> str | None:
    card = getattr(info, "card_data", None) or {}
    # huggingface_hub's ModelCardData is not a dict but implements .get().
    getter = getattr(card, "get", None)
    if callable(getter) and getter("license"):
        return getter("license")
    for tag in getattr(info, "tags", []) or []:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _languages_from(info) -> list[str]:
    card = getattr(info, "card_data", None) or {}
    getter = getattr(card, "get", None)
    lang = getter("language") if callable(getter) else None
    if isinstance(lang, str):
        return [lang]
    return list(lang) if lang else []


def to_candidate(info) -> Candidate:
    return Candidate(
        model_id=info.id,
        license=_license_from(info),
        params_b=_params_b_from_id(info.id),
        languages=_languages_from(info),
        pipeline_tag=getattr(info, "pipeline_tag", None),
        downloads=getattr(info, "downloads", 0) or 0,
        likes=getattr(info, "likes", 0) or 0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="text-classification", help="downstream task (for display)")
    ap.add_argument("--vertical", default="", help="business vertical (for display)")
    ap.add_argument("--language", default="es")
    ap.add_argument("--max-params-b", type=float, default=8.0)
    ap.add_argument(
        "--pipeline-tag",
        default="text-generation",
        help="base capability to fine-tune from",
    )
    ap.add_argument("--licenses", nargs="+", default=["apache-2.0", "mit"])
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    api = HfApi()
    raw = api.list_models(
        pipeline_tag=args.pipeline_tag,
        language=args.language,
        sort="downloads",
        direction=-1,
        limit=200,
        cardData=True,
    )
    candidates = [to_candidate(info) for info in raw]
    constraints = Constraints(
        max_params_b=args.max_params_b,
        allowed_licenses=args.licenses,
        language=args.language,
        pipeline_tag=args.pipeline_tag,
    )
    ranked = rank_candidates(candidates, constraints)

    print(f"# Scout: task={args.task} vertical={args.vertical} language={args.language}")
    max_b_str = f"max {args.max_params_b}B"
    print(
        f"# {len(ranked)} candidates within constraints ({max_b_str}, "
        f"{args.licenses})\n"
    )
    for i, s in enumerate(ranked[: args.top], 1):
        print(f"{i}. {s.candidate.model_id}  (score={s.score:.2f})")
        print(f"   {s.rationale}")


if __name__ == "__main__":
    main()
