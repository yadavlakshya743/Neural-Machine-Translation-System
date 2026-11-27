#!/usr/bin/env python3
"""Show training progress for every run in models/ (safe to run while training).

    python scripts/training_status.py            # once
    python scripts/training_status.py --watch    # refresh every 60 s (Ctrl+C to stop)

Training is finished when the log ends with "ALL DONE" (local script) and each run's status
is COMPLETED or EARLY_STOPPED with eval_test_*.json reports present.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINISHED = {"COMPLETED", "EARLY_STOPPED"}


def training_process_running() -> bool:
    """True if a scripts/train.py process exists (Windows or POSIX)."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -match 'scripts/train.py' } | Measure-Object | "
                 "Select-Object -ExpandProperty Count"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            return out not in ("", "0")
        out = subprocess.run(["pgrep", "-f", "scripts/train.py"], capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


def last_log_line(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def report() -> bool:
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    runs = sorted(glob.glob(os.path.join(ROOT, "models", "*", "metrics.json")))
    if not runs:
        print("No runs yet.")
    all_finished = bool(runs)
    for path in runs:
        run_dir = os.path.dirname(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        st, tc = data["state"], data.get("train_config", {})
        epochs = st["history"]["epochs"]
        last = epochs[-1] if epochs else {}
        evals = sorted(glob.glob(os.path.join(run_dir, "eval_test_*.json")))
        print(f"\n== {os.path.basename(run_dir)}  [{st['status']}]")
        print(f"   epoch {len(epochs)}/{tc.get('epochs')} finished | step {st['global_step']:,} | "
              f"{st['elapsed_sec'] / 3600:.1f} h trained")
        if last:
            print(f"   last epoch: dev BLEU {last.get('dev_bleu')} | chrF++ {last.get('dev_chrf2')} | "
                  f"val loss {last.get('val_loss')}")
        print(f"   best dev BLEU {st.get('best_score')} (epoch {st.get('best_epoch')})")
        for e in evals:
            r = json.load(open(e, encoding="utf-8"))
            print(f"   TEST {r['decoding']['strategy']}: BLEU {r['metrics']['bleu']} | chrF++ {r['metrics']['chrf2']}")
        if st["status"] not in FINISHED or not evals:
            all_finished = False

    log = os.path.join(ROOT, "models", "baseline_local.log")
    running = training_process_running()
    print(f"\nTraining process running: {'YES' if running else 'no'}")
    if os.path.exists(log):
        print(f"Last log line: {last_log_line(log)[:160]}")
        if "ALL DONE" in last_log_line(log):
            print("\n>>> DONE: local training and evaluation have finished. <<<")
            return True
    if not running and not all_finished:
        print("\n(!) Nothing is training and some runs are unfinished. Resume with:\n"
              "    powershell -ExecutionPolicy Bypass -File scripts\\train_baseline_local.ps1")
    return all_finished and not running


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    if not args.watch:
        report()
        return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            if report():
                break
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
