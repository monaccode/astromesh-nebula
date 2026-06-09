import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("path", ["scripts/02_train.py", "scripts/03_merge.py"])
def test_script_is_valid_python(path):
    src = Path(path).read_text(encoding="utf-8")
    ast.parse(src)  # raises SyntaxError if malformed


def test_quantize_script_targets_q4_k_m():
    src = Path("scripts/04_quantize.sh").read_text(encoding="utf-8")
    assert "Q4_K_M" in src
