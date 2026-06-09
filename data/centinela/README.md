# Centinela sentiment dataset (processed)

Spanish financial sentiment classification, derived from
[`NickyNicky/Finance_sentiment_and_topic_classification_Translation_English_to_Spanish_v1`](https://huggingface.co/datasets/NickyNicky/Finance_sentiment_and_topic_classification_Translation_English_to_Spanish_v1)
(Apache-2.0). We keep only `task_type == sentiment_analysis`, use the Spanish text
(`user_prompt_es`), and map answers to Spanish labels (`positivo`/`neutral`/`negativo`).

- Format: chat JSONL (`messages` with system/user/assistant).
- Split: stratified by label, seed 42, eval fraction 0.2, deduplicated by text (no leakage).
- PII: none (public source).
- Build: `uv run python scripts/01_build_dataset.py --dataset <id> --out data/centinela`.
