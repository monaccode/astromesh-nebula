#!/usr/bin/env bash
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
cd /mnt/d/monaccode/astromesh-nebula
PY=.venv-wsl/bin/python
"$PY" -m py_compile scripts/02_train_local.py && echo COMPILE_OK
"$PY" - <<'PY'
import torch, transformers, trl, peft, bitsandbytes, datasets
print("cuda", torch.cuda.is_available(), "| torch", torch.__version__,
      "| trl", trl.__version__, "| peft", peft.__version__,
      "| bnb", bitsandbytes.__version__, "| tf", transformers.__version__)
PY
