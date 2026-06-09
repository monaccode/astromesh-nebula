"""Create/update the public Gradio Space for the Centinela-4B demo.

    uv run python scripts/08_deploy_space.py --repo astromesh/centinela-4b-demo
"""

import argparse
import os


def deploy_space(*, api, repo_id: str, demo_dir: str) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=demo_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--demo-dir", default="demo")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    deploy_space(
        api=HfApi(token=os.environ["HF_TOKEN"]), repo_id=args.repo, demo_dir=args.demo_dir
    )
    print(f"deployed space {args.repo}")


if __name__ == "__main__":
    main()
