import json
import importlib.util
from pathlib import Path

# load the numbered script as a module
spec = importlib.util.spec_from_file_location(
    "build_dataset", str(Path("scripts/01_build_dataset.py"))
)
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)


def test_build_split_files_writes_jsonl_and_labels(tmp_path):
    rows = []
    for i in range(30):
        ans = ["positive", "neutral", "negative"][i % 3]
        rows.append(
            {
                "user_prompt_es": f"texto financiero {i}",
                "answer": ans,
                "task_type": "sentiment_analysis",
            }
        )
    out_dir = tmp_path / "centinela"
    stats = build_dataset.build_split_files(rows, str(out_dir), eval_frac=0.2, seed=42)

    train_lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    eval_lines = (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    labels = json.loads((out_dir / "labels.json").read_text(encoding="utf-8"))

    assert len(train_lines) == stats["n_train"]
    assert len(eval_lines) == stats["n_eval"]
    assert stats["n_train"] + stats["n_eval"] == 30
    assert labels == ["positivo", "neutral", "negativo"]
    # every line is a valid chat record
    rec = json.loads(train_lines[0])
    assert rec["messages"][0]["role"] == "system"
    # no leakage between splits
    train_texts = {json.loads(l)["messages"][1]["content"] for l in train_lines}
    eval_texts = {json.loads(l)["messages"][1]["content"] for l in eval_lines}
    assert train_texts.isdisjoint(eval_texts)
