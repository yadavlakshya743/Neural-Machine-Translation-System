#!/usr/bin/env python3
"""Dataset acquisition, cleaning, leakage control and SentencePiece tokenizer training.

Grounding: IIT Bombay English-Hindi Parallel Corpus (official train / dev / test split),
mirrored on Hugging Face as cfilt/iitb-english-hindi.

Generates:
  data/raw/{train,validation,test}.parquet        downloaded official splits
  data/processed/{train,dev,test}_{en,hi}.txt     normalized parallel text
  data/processed/data_report.json                 counts per cleaning stage + length stats
  data/processed/manifest.json                    sha256 of every artifact (reproducibility)
  ml/tokenization/spm_{en,hi}.model               SentencePiece BPE tokenizers (shared by
                                                  both translation directions)
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.data.preprocess import TextPreprocessor  # noqa: E402
from ml.tokenization.tokenizer import SubwordTokenizer, file_sha256  # noqa: E402

HF_BASE_URL = "https://huggingface.co/datasets/cfilt/iitb-english-hindi/resolve/main/data"
SPLIT_URLS = {
    "validation": f"{HF_BASE_URL}/validation-00000-of-00001.parquet",
    "test": f"{HF_BASE_URL}/test-00000-of-00001.parquet",
    "train": f"{HF_BASE_URL}/train-00000-of-00001.parquet",
}


def download_file(url: str, dest_path: str):
    """Downloads with an atomic rename so an interrupted download never looks complete."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[*] Already downloaded ({os.path.getsize(dest_path):,} bytes): {dest_path}")
        return
    tmp_path = dest_path + ".tmp"
    print(f"[*] Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "nmt-data-prep/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out_f.write(chunk)
                pbar.update(len(chunk))
    os.replace(tmp_path, dest_path)


def read_pairs(parquet_path: str, max_samples: int = 0):
    df = pd.read_parquet(parquet_path)
    if max_samples and len(df) > max_samples:
        df = df.iloc[:max_samples]
    en = [t.get("en", "") or "" for t in df["translation"]]
    hi = [t.get("hi", "") or "" for t in df["translation"]]
    return en, hi


def prepare_eval_split(en, hi):
    """Dev/test: normalize only; drop just empty pairs so the official benchmark is preserved."""
    out_en, out_hi = [], []
    for e, h in zip(en, hi):
        ne, nh = TextPreprocessor.normalize_english(e), TextPreprocessor.normalize_hindi(h)
        if ne and nh:
            out_en.append(ne)
            out_hi.append(nh)
    return out_en, out_hi


def prepare_train_split(en, hi, eval_sentences: set, max_len: int, max_ratio: float):
    """Train: clean/filter, deduplicate, and remove any overlap with dev/test (leakage)."""
    stats = {"raw": len(en), "rejected_by_filters": 0, "duplicates": 0, "leakage_overlap": 0}
    seen = set()
    out_en, out_hi = [], []
    for e, h in tqdm(zip(en, hi), total=len(en), desc="Cleaning train", unit="pair", mininterval=5):
        cleaned = TextPreprocessor.clean_pair(e, h, max_len=max_len, max_ratio=max_ratio)
        if cleaned is None:
            stats["rejected_by_filters"] += 1
            continue
        ce, ch = cleaned
        key = (ce, ch)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        if ce.lower() in eval_sentences or ch in eval_sentences:
            stats["leakage_overlap"] += 1
            continue
        out_en.append(ce)
        out_hi.append(ch)
    stats["kept"] = len(out_en)
    return out_en, out_hi, stats


def length_stats(lines):
    lens = np.array([len(l.split()) for l in lines]) if lines else np.zeros(1)
    return {
        "sentences": len(lines),
        "mean_words": round(float(lens.mean()), 2),
        "p50": int(np.percentile(lens, 50)),
        "p95": int(np.percentile(lens, 95)),
        "max": int(lens.max()),
    }


def save_parallel(en_lines, hi_lines, prefix: str):
    for lang, lines in (("en", en_lines), ("hi", hi_lines)):
        with open(f"{prefix}_{lang}.txt", "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    print(f"[+] Saved {prefix}_{{en,hi}}.txt ({len(en_lines):,} pairs)")


def main():
    parser = argparse.ArgumentParser(description="Acquire IIT Bombay corpus and train tokenizers.")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Use only the first N raw training rows (0 = full corpus, ~1.66M).")
    parser.add_argument("--vocab_size", type=int, default=16000,
                        help="SentencePiece BPE vocabulary size per language.")
    parser.add_argument("--max_words", type=int, default=128, help="Max words per training sentence.")
    parser.add_argument("--max_ratio", type=float, default=2.5, help="Max length ratio within a pair.")
    parser.add_argument("--spm_sample_size", type=int, default=2_000_000,
                        help="Sentences sampled to train each tokenizer.")
    parser.add_argument("--retrain_tokenizers", action="store_true",
                        help="Retrain tokenizers even if they exist. Invalidates existing checkpoints!")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--tok_dir", default="ml/tokenization")
    args = parser.parse_args()

    t0 = time.time()
    for d in (args.raw_dir, args.processed_dir, args.tok_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Download the official splits
    paths = {}
    for split, url in SPLIT_URLS.items():
        paths[split] = os.path.join(args.raw_dir, f"{split}.parquet")
        download_file(url, paths[split])

    # 2. Official dev/test, preserved as-is (normalization only)
    dev_en, dev_hi = prepare_eval_split(*read_pairs(paths["validation"]))
    test_en, test_hi = prepare_eval_split(*read_pairs(paths["test"]))
    save_parallel(dev_en, dev_hi, os.path.join(args.processed_dir, "dev"))
    save_parallel(test_en, test_hi, os.path.join(args.processed_dir, "test"))

    # 3. Train: filter, dedupe, and drop anything that also appears in dev/test
    eval_sentences = {s.lower() for s in dev_en + test_en} | set(dev_hi + test_hi)
    raw_en, raw_hi = read_pairs(paths["train"], args.max_train_samples)
    train_en, train_hi, train_stats = prepare_train_split(
        raw_en, raw_hi, eval_sentences, args.max_words, args.max_ratio
    )
    del raw_en, raw_hi
    save_parallel(train_en, train_hi, os.path.join(args.processed_dir, "train"))
    print(f"[+] Train cleaning: {train_stats}")

    # 4. Tokenizers (reused if present so existing checkpoints stay valid)
    tok_paths = {lang: os.path.join(args.tok_dir, f"spm_{lang}.model") for lang in ("en", "hi")}
    coverage = {"en": 1.0, "hi": 0.9995}
    for lang in ("en", "hi"):
        if os.path.exists(tok_paths[lang]) and not args.retrain_tokenizers:
            print(f"[*] Reusing existing tokenizer {tok_paths[lang]} (pass --retrain_tokenizers to rebuild)")
            continue
        print(f"[*] Training {lang} SentencePiece BPE tokenizer (vocab {args.vocab_size})...")
        SubwordTokenizer.train_spm(
            input_file=os.path.join(args.processed_dir, f"train_{lang}.txt"),
            model_prefix=os.path.join(args.tok_dir, f"spm_{lang}"),
            vocab_size=args.vocab_size,
            model_type="bpe",
            character_coverage=coverage[lang],
            input_sentence_size=args.spm_sample_size,
        )

    tok_en = SubwordTokenizer(tok_paths["en"])
    tok_hi = SubwordTokenizer(tok_paths["hi"])
    for tok, sample in ((tok_en, "Machine learning enables computers to learn from data."),
                        (tok_hi, "मशीन लर्निंग कंप्यूटर को डेटा से सीखने में सक्षम बनाता है।")):
        ids = tok.encode(sample)
        print(f"  {sample!r} -> {len(ids)} ids -> {tok.decode(ids)!r}")

    # 5. Report + manifest
    report = {
        "source": "cfilt/iitb-english-hindi (Hugging Face mirror of the IIT Bombay corpus)",
        "urls": SPLIT_URLS,
        "filters": {"max_words": args.max_words, "max_ratio": args.max_ratio,
                    "max_train_samples": args.max_train_samples},
        "train_cleaning": train_stats,
        "lengths": {
            "train_en": length_stats(train_en), "train_hi": length_stats(train_hi),
            "dev_en": length_stats(dev_en), "test_en": length_stats(test_en),
        },
        "vocab_size": {"en": tok_en.vocab_size, "hi": tok_hi.vocab_size},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.processed_dir, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    artifacts = [os.path.join(args.processed_dir, f"{s}_{l}.txt")
                 for s in ("train", "dev", "test") for l in ("en", "hi")] + list(tok_paths.values())
    manifest = {p.replace("\\", "/"): file_sha256(p) for p in artifacts}
    with open(os.path.join(args.processed_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Data preparation complete in {report['seconds']}s. "
          f"Train {len(train_en):,} | Dev {len(dev_en):,} | Test {len(test_en):,}")


if __name__ == "__main__":
    main()
