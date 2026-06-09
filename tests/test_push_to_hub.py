import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "push_to_hub", str(Path("scripts/06_push_to_hub.py"))
)
push_to_hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push_to_hub)


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))

    def create_tag(self, *a, **kw):
        self.calls.append(("create_tag", {"args": a, **kw}))


def test_publish_creates_repo_uploads_and_tags(tmp_path):
    # minimal artifact dirs
    (tmp_path / "merged").mkdir()
    (tmp_path / "adapter").mkdir()
    gguf = tmp_path / "model.Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")
    results = tmp_path / "results.md"
    results.write_text("| macro-F1 | 0.61 | 0.88 |", encoding="utf-8")

    api = FakeApi()
    card_path = tmp_path / "model_card" / "card.md"   # nested dir does not exist yet
    push_to_hub.publish(
        api=api,
        repo_id="astromesh/Centinela-Qwen3-4B",
        merged_dir=str(tmp_path / "merged"),
        adapter_dir=str(tmp_path / "adapter"),
        gguf_path=str(gguf),
        eval_table_path=str(results),
        card_out=str(card_path),
        tag="v0.1",
        base_model="Qwen/Qwen3-4B",
        dataset_id="NickyNicky/...",
        n_train=1000,
        n_eval=250,
        dataset_hash="abc123",
        commit="deadbeef",
    )

    names = [c[0] for c in api.calls]
    assert names.count("create_repo") == 1
    assert "upload_folder" in names
    assert "upload_file" in names
    assert "create_tag" in names
    # card rendered and written
    card = card_path.read_text(encoding="utf-8")
    assert "license: apache-2.0" in card
    assert "0.88" in card
