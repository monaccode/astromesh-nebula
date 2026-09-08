from nebula.scout import Candidate, Constraints, rank_candidates

K = Constraints(max_params_b=8.0, allowed_licenses=["apache-2.0", "mit"], language="es")


def _cand(model_id, license="apache-2.0", params_b=4.0, languages=None,
          pipeline_tag="text-generation", downloads=100, likes=10, leaderboard_score=None):
    return Candidate(model_id, license, params_b, languages or ["es"], pipeline_tag,
                     downloads, likes, leaderboard_score)


def test_drops_disallowed_license():
    out = rank_candidates([_cand("a", license="cc-by-nc-4.0")], K)
    assert out == []


def test_drops_oversized_model():
    out = rank_candidates([_cand("a", params_b=70.0)], K)
    assert out == []


def test_drops_wrong_pipeline_tag():
    out = rank_candidates([_cand("a", pipeline_tag="image-classification")], K)
    assert out == []


def test_drops_model_without_target_language():
    out = rank_candidates([_cand("a", languages=["en", "fr"])], K)
    assert out == []


def test_unknown_attributes_pass_filter():
    out = rank_candidates([_cand("a", params_b=None, languages=[], pipeline_tag=None)], K)
    assert [s.candidate.model_id for s in out] == ["a"]


def test_ranks_higher_leaderboard_first():
    a = _cand("low", leaderboard_score=0.60)
    b = _cand("high", leaderboard_score=0.90)
    out = rank_candidates([a, b], K)
    assert [s.candidate.model_id for s in out] == ["high", "low"]


def test_is_deterministic_with_tie_break_by_model_id():
    a = _cand("zzz", downloads=100, likes=10, leaderboard_score=0.5)
    b = _cand("aaa", downloads=100, likes=10, leaderboard_score=0.5)
    out = rank_candidates([a, b], K)
    assert [s.candidate.model_id for s in out] == ["aaa", "zzz"]


def test_rationale_is_populated():
    out = rank_candidates([_cand("a")], K)
    assert "apache-2.0" in out[0].rationale
