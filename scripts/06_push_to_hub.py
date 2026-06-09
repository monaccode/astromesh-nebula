"""Publish a Centinela model to the Hugging Face Hub: artifacts + card + tag.

    uv run python scripts/06_push_to_hub.py --size 4b --tag v0.1
"""

import argparse
import os

from nebula.modelcard import render_model_card


def publish(
    *,
    api,
    repo_id: str,
    merged_dir: str,
    adapter_dir: str,
    gguf_path: str,
    eval_table_path: str,
    card_out: str,
    tag: str,
    base_model: str,
    dataset_id: str,
    n_train: int,
    n_eval: int,
    dataset_hash: str,
    commit: str,
) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=merged_dir)
    api.upload_folder(repo_id=repo_id, folder_path=adapter_dir, path_in_repo="adapter")
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=gguf_path,
        path_in_repo=f"gguf/{os.path.basename(gguf_path)}",
    )

    with open(eval_table_path, encoding="utf-8") as fh:
        eval_table = fh.read().strip()

    card = render_model_card(
        repo_id=repo_id,
        base_model=base_model,
        dataset_id=dataset_id,
        eval_table=eval_table,
        n_train=n_train,
        n_eval=n_eval,
        base_model_id=base_model,
        dataset_hash=dataset_hash,
        config_path="configs/train-4b.yaml",
        commit=commit,
    )
    with open(card_out, "w", encoding="utf-8") as fh:
        fh.write(card)
    api.upload_file(repo_id=repo_id, path_or_fileobj=card_out, path_in_repo="README.md")
    api.create_tag(repo_id, tag=tag, repo_type="model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--tag", default="v0.1")
    ap.add_argument("--merged", default="export/merged")
    ap.add_argument("--adapter", default="export/adapter")
    ap.add_argument("--gguf", default="export/centinela-4b.Q4_K_M.gguf")
    ap.add_argument("--eval-table", default="eval/results/results.md")
    ap.add_argument("--dataset-hash", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--n-eval", type=int, required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    org = os.environ["HF_ORG"]
    dataset_id = (
        "NickyNicky/Finance_sentiment_and_topic_classification_"
        "Translation_English_to_Spanish_v1"
    )
    publish(
        api=HfApi(token=os.environ["HF_TOKEN"]),
        repo_id=f"{org}/Centinela-Qwen3-{args.size.upper()}",
        merged_dir=args.merged,
        adapter_dir=args.adapter,
        gguf_path=args.gguf,
        eval_table_path=args.eval_table,
        card_out="model_card/Centinela-4B.md",
        tag=args.tag,
        base_model="Qwen/Qwen3-4B",
        dataset_id=dataset_id,
        n_train=args.n_train,
        n_eval=args.n_eval,
        dataset_hash=args.dataset_hash,
        commit=args.commit,
    )
    print(f"published {org}/Centinela-Qwen3-{args.size.upper()} @ {args.tag}")


if __name__ == "__main__":
    main()
