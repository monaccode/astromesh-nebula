import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "publish_dataset", str(Path("scripts/07_publish_dataset.py"))
)
publish_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish_dataset)


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))


def test_publish_dataset_creates_dataset_repo_and_uploads(tmp_path):
    data_dir = tmp_path / "centinela"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}", encoding="utf-8")
    (data_dir / "README.md").write_text("card", encoding="utf-8")

    api = FakeApi()
    publish_dataset.publish_dataset(
        api=api, repo_id="astromesh/centinela-sentiment-es", data_dir=str(data_dir)
    )
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "upload_folder"]
    assert api.calls[0][1]["repo_type"] == "dataset"
