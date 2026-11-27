#!/usr/bin/env python3
"""Zip the code needed for Kaggle training into dist/nmt_code.zip.

Use this when the code is not on GitHub: upload the zip as a Kaggle dataset, attach it to
the notebook (Add Input) and set USE_UPLOADED_CODE = True in experiments/train_kaggle.ipynb.
Data, checkpoints, virtualenvs and node_modules are excluded; Kaggle downloads the corpus itself.
"""

import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCLUDE_DIRS = ["ml", "scripts"]
INCLUDE_FILES = ["requirements.txt", "pyproject.toml", "README.md"]
SKIP_SUFFIXES = (".pyc", ".model", ".vocab", ".pt")
# The trained tokenizers ARE included: Kaggle then reuses them (prepare_data.py never
# retrains existing tokenizers), so laptop and Kaggle models share identical tokenization.
TOKENIZERS = ["ml/tokenization/spm_en.model", "ml/tokenization/spm_hi.model"]


def main():
    out = os.path.join(ROOT, "dist", "nmt_code.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, d)):
                dirnames[:] = [n for n in dirnames if n != "__pycache__"]
                for name in filenames:
                    if name.endswith(SKIP_SUFFIXES):
                        continue
                    path = os.path.join(dirpath, name)
                    z.write(path, os.path.relpath(path, ROOT))
                    count += 1
        for name in TOKENIZERS:
            if os.path.exists(os.path.join(ROOT, name)):
                z.write(os.path.join(ROOT, name), name)
                count += 1
            else:
                print(f"[!] {name} missing: Kaggle will train its own tokenizers")
        for name in INCLUDE_FILES:
            z.write(os.path.join(ROOT, name), name)
            count += 1
    print(f"[+] Wrote {out} ({count} files, {os.path.getsize(out) / 1024:.0f} KB)")
    print("    Upload it at kaggle.com -> Datasets -> New Dataset, then Add Input in the notebook.")


if __name__ == "__main__":
    main()
