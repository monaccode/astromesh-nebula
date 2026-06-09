from pathlib import Path

import yaml


def test_release_workflow_is_manual_dispatch_with_inputs():
    wf = yaml.safe_load(Path(".github/workflows/release.yml").read_text(encoding="utf-8"))
    # PyYAML parses the bare `on:` key as boolean True
    triggers = wf.get("on", wf.get(True))
    assert "workflow_dispatch" in triggers
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert {"flavor", "tag", "ref"} <= set(inputs)


def test_eval_gate_runs_tests_and_lint():
    text = Path(".github/workflows/eval-gate.yml").read_text(encoding="utf-8")
    assert "pytest" in text and "ruff" in text


def test_run_pipeline_script_has_full_chain_in_order():
    src = Path("scripts/run_pipeline.sh").read_text(encoding="utf-8")
    order = [
        "01_build_dataset.py",
        "02_train.py",
        "03_merge.py",
        "04_quantize.sh",
        "05_evaluate.py",
        "06_push_to_hub.py",
    ]
    positions = [src.index(step) for step in order]
    assert positions == sorted(positions), "pipeline steps must appear in order"
    assert "Q4_K_M" in src
    assert "set -euo pipefail" in src
