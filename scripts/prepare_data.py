#!/usr/bin/env python3
"""Dataset acquisition, cleaning, and SentencePiece tokenizer training.

Grounding: IIT Bombay English-Hindi Parallel Corpus
Generates:
  - data/raw/ (validation.parquet, test.parquet, train.parquet)
  - data/processed/ (train, dev, test parallel .txt files)
  - ml/tokenization/spm_en.model, ml/tokenization/spm_hi.model
"""

import os
import sys
import argparse
import urllib.request
import pandas as pd
from tqdm import tqdm

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.data.preprocess import TextPreprocessor
from ml.tokenization.tokenizer import SubwordTokenizer

HF_BASE_URL = "https://huggingface.co/datasets/cfilt/iitb-english-hindi/resolve/main/data"

VAL_URL = f"{HF_BASE_URL}/validation-00000-of-00001.parquet"
TEST_URL = f"{HF_BASE_URL}/test-00000-of-00001.parquet"
TRAIN_URL = f"{HF_BASE_URL}/train-00000-of-00001.parquet"


def download_file(url: str, dest_path: str):
    """Safely downloads a file with atomic rename to prevent corrupted partial files."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[*] File already exists ({os.path.getsize(dest_path):,} bytes): {dest_path}")
        return

    tmp_path = dest_path + ".tmp"
    print(f"[*] Downloading {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                out_f.write(chunk)
                pbar.update(len(chunk))

    os.replace(tmp_path, dest_path)
    print(f"[+] Successfully downloaded: {dest_path}")


def extract_and_clean_split(parquet_path: str, max_samples: int = None, is_eval_split: bool = False):
    """Extracts parallel sentences from parquet.
    
    For evaluation splits (dev, test): Normalizes text and rejects only completely empty pairs
    to preserve 100% of the official benchmark.
    For training split: Applies length and ratio filtering to stabilize gradient updates.
    """
    print(f"[*] Reading and processing {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    if max_samples and max_samples > 0 and len(df) > max_samples:
        df = df.iloc[:max_samples]

    clean_en, clean_hi = [], []
    for item in df["translation"]:
        en_text = item.get("en", "")
        hi_text = item.get("hi", "")

        if is_eval_split:
            norm_en = TextPreprocessor.normalize_english(en_text)
            norm_hi = TextPreprocessor.normalize_hindi(hi_text)
            if norm_en and norm_hi:
                clean_en.append(norm_en)
                clean_hi.append(norm_hi)
        else:
            cleaned = TextPreprocessor.clean_pair(en_text, hi_text, max_len=128, max_ratio=2.5)
            if cleaned:
                c_en, c_hi = cleaned
                clean_en.append(c_en)
                clean_hi.append(c_hi)

    print(f"[+] Retained {len(clean_en):,} pairs from {len(df):,} rows (is_eval_split={is_eval_split}).")
    return clean_en, clean_hi


def save_parallel_corpus(en_lines, hi_lines, prefix: str):
    en_path = f"{prefix}_en.txt"
    hi_path = f"{prefix}_hi.txt"
    with open(en_path, "w", encoding="utf-8") as f_en, open(hi_path, "w", encoding="utf-8") as f_hi:
        for en, hi in zip(en_lines, hi_lines):
            f_en.write(f"{en}\n")
            f_hi.write(f"{hi}\n")
    print(f"[+] Saved: {en_path} and {hi_path} ({len(en_lines):,} lines)")


def main():
    parser = argparse.ArgumentParser(description="Acquire IIT Bombay corpus and train tokenizers.")
    parser.add_argument("--max_train_samples", type=int, default=50000,
                        help="Maximum training samples to extract (default: 50,000 for rapid baseline; 0 for all)")
    parser.add_argument("--vocab_size", type=int, default=8000,
                        help="Vocabulary size for SentencePiece BPE tokenizer (default: 8,000)")
    parser.add_argument("--skip_train_download", action="store_true",
                        help="Skip downloading the large train parquet and only use dev/test")
    args = parser.parse_args()

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    tok_dir = "ml/tokenization"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    # 1. Download official splits
    val_file = os.path.join(raw_dir, "validation.parquet")
    test_file = os.path.join(raw_dir, "test.parquet")
    train_file = os.path.join(raw_dir, "train.parquet")

    download_file(VAL_URL, val_file)
    download_file(TEST_URL, test_file)

    if not args.skip_train_download:
        download_file(TRAIN_URL, train_file)

    # 2. Extract and preserve official validation and test splits
    dev_en, dev_hi = extract_and_clean_split(val_file, is_eval_split=True)
    test_en, test_hi = extract_and_clean_split(test_file, is_eval_split=True)

    save_parallel_corpus(dev_en, dev_hi, os.path.join(processed_dir, "dev"))
    save_parallel_corpus(test_en, test_hi, os.path.join(processed_dir, "test"))

    # 3. Extract and clean training split
    if os.path.exists(train_file):
        limit = args.max_train_samples if args.max_train_samples > 0 else None
        train_en, train_hi = extract_and_clean_split(train_file, max_samples=limit, is_eval_split=False)
    else:
        print("[!] Using dev split as training data fallback since train.parquet was skipped.")
        train_en, train_hi = dev_en, dev_hi

    save_parallel_corpus(train_en, train_hi, os.path.join(processed_dir, "train"))

    # 4. Train SentencePiece Tokenizers
    print("=" * 65)
    print(f"[*] Training SentencePiece BPE Tokenizers (Vocab Size: {args.vocab_size})...")
    en_corpus_path = os.path.join(processed_dir, "train_en.txt")
    hi_corpus_path = os.path.join(processed_dir, "train_hi.txt")

    spm_en_prefix = os.path.join(tok_dir, "spm_en")
    spm_hi_prefix = os.path.join(tok_dir, "spm_hi")

    print("[*] Training English BPE Tokenizer (character_coverage=1.0)...")
    SubwordTokenizer.train_spm(
        input_file=en_corpus_path,
        model_prefix=spm_en_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
    )
    print(f"[+] English tokenizer trained: {spm_en_prefix}.model")

    print("[*] Training Hindi Devanagari BPE Tokenizer (character_coverage=0.9995)...")
    SubwordTokenizer.train_spm(
        input_file=hi_corpus_path,
        model_prefix=spm_hi_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
    )
    print(f"[+] Hindi tokenizer trained: {spm_hi_prefix}.model")

    # 5. Verify Tokenizers
    tok_en = SubwordTokenizer(model_path=f"{spm_en_prefix}.model")
    tok_hi = SubwordTokenizer(model_path=f"{spm_hi_prefix}.model")

    sample_en = "Machine learning enables computers to learn representations from data."
    sample_hi = "मशीन लर्निंग कंप्यूटर को डेटा से सीखने में सक्षम बनाता है।"

    en_ids = tok_en.encode(sample_en)
    hi_ids = tok_hi.encode(sample_hi)

    print("=" * 65)
    print("[*] Verification Demonstration:")
    print(f"  English Input:  {sample_en}")
    print(f"  Encoded IDs:    {en_ids[:12]}... (length: {len(en_ids)})")
    print(f"  Decoded Back:   {tok_en.decode(en_ids)}")
    print(f"  Hindi Input:    {sample_hi}")
    print(f"  Encoded IDs:    {hi_ids[:12]}... (length: {len(hi_ids)})")
    print(f"  Decoded Back:   {tok_hi.decode(hi_ids)}")
    print("=" * 65)
    print("[+] Step 1 (Data Preprocessing & Tokenizer Training) COMPLETE!")


if __name__ == "__main__":
    main()
