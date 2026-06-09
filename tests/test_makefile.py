from pathlib import Path


def test_makefile_has_pipeline_targets():
    mk = Path("Makefile").read_text(encoding="utf-8")
    for target in ["dataset:", "train:", "eval:", "quantize:", "publish:", "test:"]:
        assert target in mk
