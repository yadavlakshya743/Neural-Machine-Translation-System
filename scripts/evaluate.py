#!/usr/bin/env python3
"""CLI evaluation script for computing SacreBLEU and chrF++ benchmarks on test data.

Grounding: Official IIT Bombay Parallel Test Corpus (2,507 sentences)
Metrics: SacreBLEU (word n-gram precision) and chrF++ (character n-gram F-score)
"""

import os
import sys
import json
import argparse
import time
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.transformer.model import TransformerSeq2Seq
from ml.tokenization.tokenizer import SubwordTokenizer
from ml.inference.translator import Translator
from ml.evaluation.evaluator import TranslationEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate NMT model on official test benchmark.")
    parser.add_argument("--model_path", type=str, default="models/best_transformer_en_hi.pt",
                        help="Path to trained checkpoint .pt file")
    parser.add_argument("--src_tok", type=str, default="ml/tokenization/spm_en.model",
                        help="Path to source SentencePiece model")
    parser.add_argument("--tgt_tok", type=str, default="ml/tokenization/spm_hi.model",
                        help="Path to target SentencePiece model")
    parser.add_argument("--test_src", type=str, default="data/processed/test_en.txt",
                        help="Path to official test source text file")
    parser.add_argument("--test_tgt", type=str, default="data/processed/test_hi.txt",
                        help="Path to official test target reference file")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of test sentences to evaluate (default: 200 for rapid baseline; 0 for all 2,507)")
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "beam_search"],
                        help="Decoding strategy: greedy or beam_search")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam width for beam search")
    parser.add_argument("--output_json", type=str, default="experiments/baseline_experiment.json",
                        help="Output path for evaluation report JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("      OFFICIAL BENCHMARK EVALUATION (SacreBLEU & chrF++)      ")
    print("=" * 70)
    print(f"[*] Checkpoint:     {args.model_path}")
    print(f"[*] Device:         {device.upper()}")
    print(f"[*] Test Source:    {args.test_src}")
    print(f"[*] Test Reference: {args.test_tgt}")
    print(f"[*] Strategy:       {args.strategy}")

    # 1. Verify and Load Tokenizers
    if not os.path.exists(args.src_tok) or not os.path.exists(args.tgt_tok):
        raise FileNotFoundError("SentencePiece tokenizer models not found. Run prepare_data.py first.")

    src_tok = SubwordTokenizer(model_path=args.src_tok)
    tgt_tok = SubwordTokenizer(model_path=args.tgt_tok)

    # 2. Verify and Load Checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}. Train a baseline model first.")

    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Extract architecture dimensions from metadata if present
    d_model = ckpt.get("d_model", 256)
    n_heads = ckpt.get("n_heads", 8)
    n_encoder_layers = ckpt.get("n_encoder_layers", 4)
    n_decoder_layers = ckpt.get("n_decoder_layers", 4)
    d_ff = ckpt.get("d_ff", 1024)

    model = TransformerSeq2Seq(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        pad_idx=SubwordTokenizer.PAD_ID,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[+] Loaded Model Checkpoint (Epoch: {ckpt.get('epoch', 'N/A')}, Train Loss: {ckpt.get('train_loss', 'N/A')})")

    # 3. Load Test Data
    with open(args.test_src, "r", encoding="utf-8") as f_src, open(args.test_tgt, "r", encoding="utf-8") as f_tgt:
        sources = [line.strip() for line in f_src if line.strip()]
        references = [line.strip() for line in f_tgt if line.strip()]

    min_pairs = min(len(sources), len(references))
    sources = sources[:min_pairs]
    references = references[:min_pairs]

    total_available = len(sources)
    if args.num_samples and args.num_samples > 0 and args.num_samples < total_available:
        sources = sources[:args.num_samples]
        references = references[:args.num_samples]

    print(f"[+] Evaluating on {len(sources):,} of {total_available:,} official benchmark sentences.")

    # 4. Generate Hypotheses
    translator = Translator(
        model=model,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        device=device,
        default_strategy=args.strategy,
        beam_size=args.beam_size,
    )

    hypotheses = []
    latencies = []
    print("[*] Translating test set...")
    start_eval_time = time.time()

    for src_text in tqdm(sources, desc="Evaluating", unit="sent"):
        res = translator.translate(src_text, strategy=args.strategy, beam_size=args.beam_size, max_len=128)
        hypotheses.append(res["translated_text"])
        latencies.append(res["latency_ms"])

    total_eval_time = time.time() - start_eval_time
    avg_latency = sum(latencies) / max(1, len(latencies))

    # 5. Compute BLEU and chrF++
    ref_list = [[r] for r in references]
    scores = TranslationEvaluator.compute_metrics(hypotheses, ref_list)
    bleu_score = scores.get("bleu", 0.0)
    chrf_score = scores.get("chrf2", 0.0)

    # 6. Save JSON Report
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    report = {
        "experiment_name": "Experiment 1 - Baseline Transformer",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": args.model_path,
        "device": device,
        "checkpoint_epoch": ckpt.get("epoch", 1),
        "checkpoint_train_loss": ckpt.get("train_loss", None),
        "checkpoint_val_loss": ckpt.get("val_loss", None),
        "test_samples_evaluated": len(sources),
        "decoding_strategy": args.strategy,
        "beam_size": args.beam_size if args.strategy == "beam_search" else 1,
        "metrics": {
            "bleu": bleu_score,
            "chrf2": chrf_score,
        },
        "performance": {
            "total_eval_time_sec": round(total_eval_time, 2),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "sample_translations": [
            {
                "index": i + 1,
                "source": sources[i],
                "reference": references[i],
                "hypothesis": hypotheses[i],
            }
            for i in range(min(5, len(sources)))
        ],
    }

    with open(args.output_json, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("                   EVALUATION RESULTS REPORT                  ")
    print("=" * 70)
    print(f"  SacreBLEU Score:       {bleu_score}")
    print(f"  chrF++ Score:          {chrf_score}")
    print(f"  Avg Latency:           {avg_latency:.2f} ms/sentence")
    print(f"  Total Evaluation Time: {total_eval_time:.1f} s")
    print(f"  Detailed Report:       {args.output_json}")
    print("-" * 70)
    print("Sample Output Comparison:")
    for item in report["sample_translations"][:3]:
        print(f"  [{item['index']}] Source:    {item['source']}")
        print(f"      Reference: {item['reference']}")
        print(f"      Model:     \"{item['hypothesis']}\"")
        print()
    print("=" * 70)
    print("[+] Evaluation Benchmark Complete!")


if __name__ == "__main__":
    main()
