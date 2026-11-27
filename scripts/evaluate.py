#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the official IIT Bombay test (or dev) split.

Reports SacreBLEU and chrF++ with metric signatures, decoding configuration, latency and
sample translations. Writes <run_dir>/eval_<split>_<strategy>.json and the hypotheses file.

Example:
  python scripts/evaluate.py --checkpoint models/transformer_base_en-hi/best.pt --strategy beam_search
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.evaluation.evaluator import TranslationEvaluator  # noqa: E402
from ml.inference.loader import load_translator  # noqa: E402


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    p = argparse.ArgumentParser(description="Evaluate an NMT checkpoint with BLEU and chrF++.")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt written by scripts/train.py")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--strategy", default="beam_search", choices=["greedy", "beam_search"])
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.6, help="Length penalty for beam search")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=0, help="0 = whole split (2,507 test sentences)")
    p.add_argument("--bleu_tokenize", default="13a")
    p.add_argument("--output_dir", default=None, help="Default: the checkpoint's directory")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    translator, meta = load_translator(args.checkpoint, device=device,
                                       beam_size=args.beam_size, alpha=args.alpha)
    src, tgt = meta["src_lang"], meta["tgt_lang"]

    sources = read_lines(os.path.join(args.processed_dir, f"{args.split}_{src}.txt"))
    references = read_lines(os.path.join(args.processed_dir, f"{args.split}_{tgt}.txt"))
    assert len(sources) == len(references), "source/reference line counts differ"
    total = len(sources)
    if args.num_samples and args.num_samples < total:
        sources, references = sources[: args.num_samples], references[: args.num_samples]

    print(f"[*] {meta.get('run_name')} | {src}->{tgt} | {args.split} {len(sources):,}/{total:,} "
          f"| {args.strategy}" + (f" k={args.beam_size} alpha={args.alpha}" if args.strategy == "beam_search" else ""))

    t0 = time.time()
    ids = translator.translate_batch(sources, strategy=args.strategy, beam_size=args.beam_size,
                                     alpha=args.alpha, max_len=args.max_len, batch_size=args.batch_size)
    hypotheses = [translator.detokenize(x) for x in ids]
    elapsed = time.time() - t0

    scores = TranslationEvaluator.compute_metrics(hypotheses, [[r] for r in references],
                                                  bleu_tokenize=args.bleu_tokenize)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.split}_{args.strategy}" + (f"_k{args.beam_size}" if args.strategy == "beam_search" else "")
    hyp_path = os.path.join(out_dir, f"hyp_{tag}.{tgt}.txt")
    with open(hyp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(hypotheses) + "\n")

    report = {
        "run_name": meta.get("run_name"),
        "checkpoint": args.checkpoint,
        "direction": f"{src}->{tgt}",
        "split": args.split,
        "samples_evaluated": len(sources),
        "split_size": total,
        "decoding": {"strategy": args.strategy, "beam_size": args.beam_size if args.strategy == "beam_search" else 1,
                     "alpha": args.alpha if args.strategy == "beam_search" else None, "max_len": args.max_len},
        "metrics": scores,
        "performance": {"total_sec": round(elapsed, 2),
                        "ms_per_sentence": round(1000 * elapsed / max(1, len(sources)), 2),
                        "device": device},
        "model": {"parameters": meta.get("total_parameters"), "train_pairs": meta.get("train_pairs"),
                  "config": meta.get("config", {}).get("model")},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": [{"source": s, "reference": r, "hypothesis": h}
                    for s, r, h in list(zip(sources, references, hypotheses))[:20]],
        "note": "Automatic metrics are evidence of quality, not proof of semantic correctness.",
    }
    report_path = os.path.join(out_dir, f"eval_{tag}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[+] BLEU {scores['bleu']} | chrF++ {scores['chrf2']} | "
          f"{report['performance']['ms_per_sentence']} ms/sentence | {elapsed:.1f}s total")
    print(f"    {scores['bleu_signature']}")
    print(f"[+] Report: {report_path}")
    for item in report["samples"][:3]:
        print(f"  SRC {item['source']}\n  REF {item['reference']}\n  HYP {item['hypothesis']}\n")


if __name__ == "__main__":
    main()
