#!/usr/bin/env python3
"""Create / update the Hugging Face Space that hosts the app.

First time (after `hf auth login` with a *write* token):
    python scripts/setup_space.py --user <hf-username>

This creates <user>/nmt-translator (Docker Space) and <user>/nmt-en-hi-models (model repo),
sets the Space's SECRET_KEY secret and HF_MODEL_REPO variable, publishes the current
checkpoints and uploads the code. Later code updates are pushed automatically by
.github/workflows/deploy-space.yml, which runs:
    python scripts/setup_space.py --code-only --space <user>/nmt-translator
"""

import argparse
import os
import secrets
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIRS = ["ml", "backend", "frontend"]
CODE_FILES = ["requirements.txt", "deploy/requirements-serve.txt", "scripts/fetch_models.py"]
SKIP = shutil.ignore_patterns("node_modules", "dist", "__pycache__", "*.pyc", "*.model", "*.vocab",
                              "*.pt", ".env", "*.db")


def assemble(stage: str) -> None:
    """Copy exactly what the Space image needs, plus its Dockerfile and README."""
    for d in CODE_DIRS:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(stage, d), ignore=SKIP)
    for f in CODE_FILES:
        os.makedirs(os.path.dirname(os.path.join(stage, f)) or stage, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, f), os.path.join(stage, f))
    shutil.copy2(os.path.join(ROOT, "deploy", "Dockerfile.web"), os.path.join(stage, "Dockerfile"))
    shutil.copy2(os.path.join(ROOT, "deploy", "space", "README.md"), os.path.join(stage, "README.md"))


def upload_code(api, space: str, message: str) -> None:
    with tempfile.TemporaryDirectory() as stage:
        assemble(stage)
        # delete_patterns="*" mirrors the folder: files removed from the repo disappear from the Space.
        api.upload_folder(folder_path=stage, repo_id=space, repo_type="space",
                          commit_message=message, delete_patterns="*")
    print(f"[+] Code uploaded to Space {space}; it now rebuilds (first build ~5-10 min).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", help="Hugging Face username (first-time setup)")
    p.add_argument("--space-name", default="nmt-translator")
    p.add_argument("--model-repo-name", default="nmt-en-hi-models")
    p.add_argument("--space", help="Full Space id, e.g. user/nmt-translator (with --code-only)")
    p.add_argument("--code-only", action="store_true", help="Only upload code (used by GitHub Actions)")
    p.add_argument("--message", default="Deploy from local machine")
    args = p.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    if args.code_only:
        if not args.space:
            sys.exit("--code-only needs --space user/space-name")
        upload_code(api, args.space, args.message)
        return

    if not args.user:
        sys.exit("Give --user <your Hugging Face username>")
    who = api.whoami()["name"]
    if who != args.user:
        print(f"[!] Logged in as {who!r}, not {args.user!r}; repos will be created under {args.user!r} "
              "only if this token has access to it.")
    space = f"{args.user}/{args.space_name}"
    model_repo = f"{args.user}/{args.model_repo_name}"

    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_secret(space, "SECRET_KEY", secrets.token_urlsafe(32))
    api.add_space_variable(space, "HF_MODEL_REPO", model_repo)
    print(f"[+] Space {space} ready (secret SECRET_KEY, variable HF_MODEL_REPO={model_repo})")

    sys.path.insert(0, os.path.dirname(__file__))
    from publish_models import publish
    publish(model_repo, space=None)

    upload_code(api, space, "Initial deployment")
    print("\nDeployment links:")
    print(f"  Space page : https://huggingface.co/spaces/{space}")
    print(f"  Direct app : https://{args.user.lower()}-{args.space_name.lower()}.hf.space")
    print(f"  API docs   : https://{args.user.lower()}-{args.space_name.lower()}.hf.space/docs")


if __name__ == "__main__":
    main()
