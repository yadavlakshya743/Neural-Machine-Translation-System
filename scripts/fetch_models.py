#!/usr/bin/env python3
"""Download the published checkpoints into ./models at container startup.

Reads HF_MODEL_REPO (e.g. "username/nmt-en-hi-models"). Never fails the startup: if the
download is impossible the API still starts and /ready reports that no model is loaded.
"""

import os
import sys

PATTERNS = ["*/best.pt", "*/metrics.json", "*/config.yaml", "*/spm_*.model", "*/eval_*.json",
            "serving.json", "data_report.json"]


def main() -> int:
    repo = os.environ.get("HF_MODEL_REPO", "").strip()
    if not repo:
        print("[fetch_models] HF_MODEL_REPO not set; starting without downloading models.")
        return 0
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo, repo_type="model", local_dir="models",
                                 allow_patterns=PATTERNS, token=os.environ.get("HF_TOKEN") or None)
        print(f"[fetch_models] Downloaded {repo} into {path}")
    except Exception as e:  # network/permission problems must not stop the API from starting
        print(f"[fetch_models] Could not download {repo}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
