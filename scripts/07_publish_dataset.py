"""Publish the processed Spanish split + dataset card (public source, no PII).

    uv run python scripts/07_publish_dataset.py --repo astromesh/centinela-sentiment-es
"""

import argparse
import os


def publish_dataset(*, api, repo_id: str, data_dir: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=data_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-dir", default="data/centinela")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    publish_dataset(
        api=HfApi(token=os.environ["HF_TOKEN"]),
        repo_id=args.repo,
        data_dir=args.data_dir,
    )
    print(f"published dataset {args.repo}")


if __name__ == "__main__":
    main()
