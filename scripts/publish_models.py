#!/usr/bin/env python3
"""Upload the latest trained checkpoints to a Hugging Face model repo and restart the Space.

Run this whenever training has produced a better model (it is safe while training runs):

    python scripts/publish_models.py --repo <user>/nmt-en-hi-models --space <user>/nmt-translator

For each direction the run with the highest best dev BLEU is selected and written to
serving.json, which the deployed backend uses to decide which checkpoint to serve.
Requires a one-time `hf auth login` (or HF_TOKEN in the environment).
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_FILES = ["best.pt", "metrics.json", "config.yaml", "spm_en.model", "spm_hi.model"]


def discover_runs(models_dir):
    runs = []
    for best in glob.glob(os.path.join(models_dir, "*", "best.pt")):
        run_dir = os.path.dirname(best)
        name = os.path.basename(run_dir)
        if name.startswith("smoke"):
            continue
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        runs.append({
            "name": name,
            "dir": run_dir,
            "direction": data["meta"]["direction"],
            "best": data["state"].get("best_score") or 0.0,
            "status": data["state"].get("status"),
            "epochs": len(data["state"]["history"]["epochs"]),
        })
    return runs


def publish(repo, space=None, models_dir=None, private=False):
    from huggingface_hub import HfApi

    models_dir = models_dir or os.path.join(ROOT, "models")
    runs = discover_runs(models_dir)
    if not runs:
        sys.exit("No runs with best.pt + metrics.json in models/. Train first.")

    serving = {}
    for r in sorted(runs, key=lambda r: r["best"]):
        serving[r["direction"]] = f"{r['name']}/best.pt"  # highest dev BLEU wins per direction

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as stage:
        # Copy first: the upload then never holds files the training process may replace.
        for r in runs:
            dst = os.path.join(stage, r["name"])
            os.makedirs(dst)
            for name in RUN_FILES + [os.path.basename(p) for p in glob.glob(os.path.join(r["dir"], "eval_*.json"))]:
                src = os.path.join(r["dir"], name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dst, name))
        report = os.path.join(ROOT, "data", "processed", "data_report.json")
        if os.path.exists(report):
            shutil.copy2(report, os.path.join(stage, "data_report.json"))
        with open(os.path.join(stage, "serving.json"), "w", encoding="utf-8") as f:
            json.dump(serving, f, indent=2)

        summary = ", ".join(f"{r['name']} ({r['direction']}, {r['epochs']} ep, dev BLEU {r['best']})" for r in runs)
        api.upload_folder(folder_path=stage, repo_id=repo, repo_type="model",
                          commit_message=f"Publish checkpoints: {summary}")

    print(f"[+] Uploaded to https://huggingface.co/{repo}")
    for r in runs:
        mark = "  <- serving" if serving.get(r["direction"]) == f"{r['name']}/best.pt" else ""
        print(f"    {r['name']:<28} {r['direction']}  epochs {r['epochs']:<3} dev BLEU {r['best']}  [{r['status']}]{mark}")
    if space:
        api.restart_space(space)
        print(f"[+] Restarted Space {space}; it downloads the new models on startup (~1-2 min).")
    return serving


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="Model repo id, e.g. username/nmt-en-hi-models")
    p.add_argument("--space", default=None, help="Space id to restart afterwards, e.g. username/nmt-translator")
    p.add_argument("--private", action="store_true", help="Private model repo (then the Space needs an HF_TOKEN secret)")
    args = p.parse_args()
    publish(args.repo, args.space, private=args.private)


if __name__ == "__main__":
    main()
