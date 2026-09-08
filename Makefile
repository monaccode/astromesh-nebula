SIZE ?= 4b
DATASET ?= NickyNicky/Finance_sentiment_and_topic_classification_Translation_English_to_Spanish_v1
TASK ?= text-classification
VERTICAL ?= finanzas
MAXB ?= 8

.PHONY: test lint dataset train eval quantize merge publish space catalog scout lock

test:
	uv run pytest -v

lint:
	uv run ruff check nebula/ tests/ scripts/

scout:
	uv run python scripts/00_scout.py --task $(TASK) --vertical $(VERTICAL) \
		--language es --max-params-b $(MAXB) --top 5

dataset:
	uv run python scripts/01_build_dataset.py --dataset $(DATASET) --out data/centinela

train:
	hf jobs uv run --flavor a10g-large scripts/02_train.py -- \
		--config configs/train-$(SIZE).yaml --train data/centinela/train.jsonl --out export/adapter

merge:
	uv run python scripts/03_merge.py --base Qwen/Qwen3-4B --adapter export/adapter --out export/merged

quantize:
	bash scripts/04_quantize.sh export/merged export/centinela-$(SIZE).Q4_K_M.gguf

eval:
	uv run python scripts/05_evaluate.py --base Qwen/Qwen3-4B --tuned export/merged \
		--eval data/centinela/eval.jsonl --out eval/results/results.md

publish:
	uv run python scripts/06_push_to_hub.py --size $(SIZE) --tag $(TAG) --gguf export/centinela-$(SIZE).Q4_K_M.gguf

space:
	uv run python scripts/08_deploy_space.py --repo $(HF_ORG)/centinela-$(SIZE)-demo

catalog:
	uv run python scripts/09_catalog.py compile \
		--models catalog/centinela.yaml --datasets catalog/datasets.yaml --out catalog.lock.json

lock:  ## Compile the catalog lock shipped inside the wheel
	uv run python scripts/09_catalog.py compile \
		--models catalog/centinela.yaml --datasets catalog/datasets.yaml \
		--out nebula/catalog.lock.json