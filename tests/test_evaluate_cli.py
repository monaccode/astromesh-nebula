import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "evaluate_cli", str(Path("scripts/05_evaluate.py"))
)
evaluate_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_cli)


def test_format_results_table_includes_both_columns():
    base = {"accuracy": 0.61, "macro_f1": 0.60, "invalid_rate": 0.10}
    tuned = {"accuracy": 0.90, "macro_f1": 0.88, "invalid_rate": 0.0}
    table = evaluate_cli.format_results_table(base, tuned)
    assert "macro-F1" in table
    assert "0.60" in table and "0.88" in table
    assert "Base" in table and "Centinela-4B" in table


def test_gate_passes_requires_threshold_and_beating_base():
    base = {"macro_f1": 0.60}
    tuned_good = {"macro_f1": 0.88}
    tuned_bad = {"macro_f1": 0.80}
    assert evaluate_cli.gate_passes(tuned_good, base, threshold=0.85, must_beat_base=True)
    assert not evaluate_cli.gate_passes(tuned_bad, base, threshold=0.85, must_beat_base=True)
    # beats base but below threshold -> still fails
    assert not evaluate_cli.gate_passes({"macro_f1": 0.70}, base, threshold=0.85, must_beat_base=True)
