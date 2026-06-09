from nebula.data import row_to_chat_record, stratified_split, record_label
from nebula.labels import SYSTEM_PROMPT_ES


def _row(es, answer, task="sentiment_analysis"):
    return {
        "user_prompt_en": "ignored",
        "user_prompt_es": es,
        "answer": answer,
        "system_prompt": "ignored english prompt",
        "task_type": task,
    }


def test_row_maps_to_spanish_chat_record():
    rec = row_to_chat_record(_row("La acción subió fuerte.", "positive"))
    assert rec["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT_ES}
    assert rec["messages"][1] == {"role": "user", "content": "La acción subió fuerte."}
    assert rec["messages"][2] == {"role": "assistant", "content": "positivo"}


def test_row_skips_non_sentiment_tasks():
    assert row_to_chat_record(_row("texto", "positive", task="topic_classification")) is None


def test_row_skips_unknown_answer():
    assert row_to_chat_record(_row("texto", "mixed")) is None


def test_row_skips_empty_spanish_text():
    assert row_to_chat_record(_row("   ", "positive")) is None


def test_record_label_returns_assistant_content():
    rec = row_to_chat_record(_row("texto", "negative"))
    assert record_label(rec) == "negativo"


def test_stratified_split_is_deterministic_and_balanced():
    records = []
    for i in range(30):
        ans = ["positive", "neutral", "negative"][i % 3]
        records.append(row_to_chat_record(_row(f"texto {i}", ans)))
    train1, eval1 = stratified_split(records, eval_frac=0.2, seed=42)
    train2, eval2 = stratified_split(records, eval_frac=0.2, seed=42)
    # deterministic
    assert [record_label(r) for r in eval1] == [record_label(r) for r in eval2]
    # ~20% held out, every class represented in eval
    assert len(eval1) == 6
    assert {record_label(r) for r in eval1} == {"positivo", "neutral", "negativo"}
    # no leakage: no eval text appears in train
    train_texts = {r["messages"][1]["content"] for r in train1}
    eval_texts = {r["messages"][1]["content"] for r in eval1}
    assert train_texts.isdisjoint(eval_texts)
