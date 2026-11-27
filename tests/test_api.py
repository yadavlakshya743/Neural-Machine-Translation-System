"""API integration tests (PRD TEST-001..011) against a tiny real checkpoint + tokenizers."""

import json
import os
import shutil

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from ml.tokenization.tokenizer import SubwordTokenizer, file_sha256
from ml.transformer.model import TransformerSeq2Seq

EN = ["i am going to college", "how are you", "the weather is nice today", "india is a big country",
      "machine learning is useful", "she reads a book", "we play cricket", "this is a test sentence"]
HI = ["मैं कॉलेज जा रहा हूँ", "आप कैसे हैं", "आज मौसम अच्छा है", "भारत एक बड़ा देश है",
      "मशीन लर्निंग उपयोगी है", "वह एक किताब पढ़ती है", "हम क्रिकेट खेलते हैं", "यह एक परीक्षण वाक्य है"]


def _train_tiny_spm(lines, prefix):
    import sentencepiece as spm
    with open(prefix + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines * 20))
    spm.SentencePieceTrainer.train(
        input=prefix + ".txt", model_prefix=prefix, vocab_size=60, model_type="char",
        hard_vocab_limit=False, pad_id=0, bos_id=1, eos_id=2, unk_id=3, minloglevel=2,
    )
    return prefix + ".model"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    run_dir = root / "models" / "tiny_en-hi"
    run_dir.mkdir(parents=True)
    en_model = _train_tiny_spm(EN, str(root / "spm_en"))
    hi_model = _train_tiny_spm(HI, str(root / "spm_hi"))
    shutil.copy(en_model, run_dir / "spm_en.model")
    shutil.copy(hi_model, run_dir / "spm_hi.model")
    src_tok, tgt_tok = SubwordTokenizer(str(run_dir / "spm_en.model")), SubwordTokenizer(str(run_dir / "spm_hi.model"))

    torch.manual_seed(0)
    model = TransformerSeq2Seq(src_tok.vocab_size, tgt_tok.vocab_size, d_model=32, n_heads=4,
                               n_encoder_layers=1, n_decoder_layers=1, d_ff=64, dropout=0.0, max_len=256)
    meta = {
        "run_name": "tiny_en-hi", "direction": "en-hi", "src_lang": "en", "tgt_lang": "hi",
        "max_src_len": 128, "total_parameters": sum(p.numel() for p in model.parameters()),
        "train_pairs": len(EN), "model_config": model.config,
        "tokenizers": {
            "src": {"path": "ml/tokenization/spm_en.model", "sha256": file_sha256(str(run_dir / "spm_en.model"))},
            "tgt": {"path": "ml/tokenization/spm_hi.model", "sha256": file_sha256(str(run_dir / "spm_hi.model"))},
        },
    }
    ckpt = run_dir / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, ckpt)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"meta": {**meta, "config": {"model": {"d_model": 32}}}, "train_config": {},
                   "state": {"status": "COMPLETED", "global_step": 10, "best_score": 1.5, "best_epoch": 1,
                             "elapsed_sec": 60, "history": {"train": [{"step": 10, "loss": 3.0}],
                                                            "epochs": [{"epoch": 1, "dev_bleu": 1.5}]}}}, f)

    saved = {k: getattr(settings, k) for k in ("MODEL_CHECKPOINT_PATH", "MODEL_CHECKPOINT_PATH_HI_EN",
                                               "DATABASE_URL", "MODELS_DIR", "DEVICE")}
    settings.MODEL_CHECKPOINT_PATH = str(ckpt)
    settings.MODEL_CHECKPOINT_PATH_HI_EN = str(root / "missing" / "best.pt")
    settings.DATABASE_URL = "sqlite:///" + str(root / "test.db").replace("\\", "/")
    settings.MODELS_DIR = str(root / "models")
    settings.DEVICE = "cpu"

    from backend.app.main import app
    with TestClient(app) as c:
        c.ckpt_path = str(ckpt)
        yield c
    for k, v in saved.items():
        setattr(settings, k, v)


def _auth(client, username="alice", password="password123"):
    client.post("/api/v1/auth/register", json={"username": username, "password": password})
    r = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _translate(client, headers, **overrides):
    body = {"source_text": "how are you", "source_lang": "en", "target_lang": "hi",
            "decoding_strategy": "greedy", **overrides}
    return client.post("/api/v1/translations", json=body, headers=headers)


# TEST-011 health / readiness distinguish app vs model
def test_health_and_readiness(client):
    assert client.get("/api/v1/health").json()["status"] == "healthy"
    r = client.get("/api/v1/ready")
    assert r.status_code == 200
    assert r.json()["loaded_directions"] == ["en-hi"] and r.json()["missing_directions"] == ["hi-en"]


def test_auth_register_login_and_errors(client):
    assert client.post("/api/v1/auth/register", json={"username": "bob", "password": "password123"}).status_code == 201
    dup = client.post("/api/v1/auth/register", json={"username": "bob", "password": "password123"})
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "USERNAME_TAKEN"
    bad = client.post("/api/v1/auth/login", json={"username": "bob", "password": "wrong-password"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "INVALID_CREDENTIALS"
    short = client.post("/api/v1/auth/register", json={"username": "x", "password": "short"})
    assert short.status_code == 422
    headers = _auth(client, "bob")
    assert client.get("/api/v1/auth/me", headers=headers).json()["username"] == "bob"


def test_translation_requires_login(client):
    r = _translate(client, {})
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED"
    r = _translate(client, {"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


# TEST-001 valid request, TEST-005 fixed-checkpoint inference, TEST-007 beam search bounded
def test_translation_success_greedy_and_beam(client):
    headers = _auth(client)
    r = _translate(client, headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["model_version"] == "tiny_en-hi" and data["source_lang"] == "en"
    assert isinstance(data["translated_text"], str) and data["tokens_generated"] <= 128
    again = _translate(client, headers)
    assert again.json()["translated_text"] == data["translated_text"]  # deterministic
    beam = _translate(client, headers, decoding_strategy="beam_search", beam_size=3, max_length=20)
    assert beam.status_code == 200 and beam.json()["beam_size"] == 3
    assert beam.json()["tokens_generated"] <= 20
    assert r.headers.get("X-Request-ID")


# TEST-002 invalid direction, TEST-003 empty/oversized input
@pytest.mark.parametrize("overrides,status,code", [
    ({"source_lang": "en", "target_lang": "en"}, 400, "INVALID_LANGUAGE_DIRECTION"),
    ({"source_lang": "fr"}, 422, "VALIDATION_ERROR"),
    ({"source_text": "   "}, 422, "VALIDATION_ERROR"),
    ({"source_text": "a" * 1001}, 422, "VALIDATION_ERROR"),
    ({"beam_size": 11}, 422, "VALIDATION_ERROR"),
])
def test_input_validation(client, overrides, status, code):
    r = _translate(client, _auth(client), **overrides)
    assert r.status_code == status and r.json()["error"]["code"] == code


# TEST-009 model unavailable -> graceful error
def test_missing_direction_returns_503(client):
    r = _translate(client, _auth(client), source_text="नमस्ते", source_lang="hi", target_lang="en")
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_AVAILABLE"


# TEST-008 users only see their own history
def test_history_is_private(client):
    alice = _auth(client, "alice2")
    carol = _auth(client, "carol")
    tid = _translate(client, alice).json()["request_id"]
    page = client.get("/api/v1/translations", headers=alice).json()
    assert page["total"] >= 1 and page["items"][0]["request_id"] == tid
    assert client.get(f"/api/v1/translations/{tid}", headers=alice).status_code == 200
    assert client.get("/api/v1/translations", headers=carol).json()["total"] == 0
    r = client.get(f"/api/v1/translations/{tid}", headers=carol)
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    assert client.get("/api/v1/translations").status_code == 401


def test_models_and_evaluations(client):
    m = client.get("/api/v1/models/active").json()
    assert m["loaded"][0]["run_name"] == "tiny_en-hi" and m["loaded"][0]["parameters"] > 0
    assert m["loaded"][0]["tokenizer_sha256"]["src"] and m["missing_directions"] == ["hi-en"]
    ev = client.get("/api/v1/evaluations").json()
    assert ev["experiments"][0]["run_name"] == "tiny_en-hi"
    assert ev["experiments"][0]["epochs"][0]["dev_bleu"] == 1.5


def test_inference_failure_hides_internals(client, monkeypatch):
    from backend.app.services.translation_service import translation_service

    def boom(**kwargs):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(translation_service.translators["en-hi"], "translate", boom)
    r = _translate(client, _auth(client))
    assert r.status_code == 500 and r.json()["error"]["code"] == "INFERENCE_FAILED"
    assert "secret" not in r.text and "Traceback" not in r.text


# TEST-006 tokenizer/checkpoint mismatch blocks loading
def test_tokenizer_mismatch_blocks_load(client, tmp_path):
    from ml.inference.loader import CheckpointMismatchError, load_checkpoint

    ckpt = torch.load(client.ckpt_path, weights_only=False)
    ckpt["meta"]["tokenizers"]["src"]["sha256"] = "0" * 64
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    for name in ("spm_en.model", "spm_hi.model"):
        shutil.copy(os.path.join(os.path.dirname(client.ckpt_path), name), bad_dir / name)
    torch.save(ckpt, bad_dir / "best.pt")
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(str(bad_dir / "best.pt"))


def test_checkpoint_with_windows_paths_loads(client, tmp_path):
    """Checkpoints trained on Windows store 'ml\tokenization\spm_en.model'; Linux servers must
    still find the tokenizer saved next to the checkpoint (regression: deploy failed on this)."""
    from ml.inference.loader import load_checkpoint

    ckpt = torch.load(client.ckpt_path, weights_only=False)
    for side, name in (("src", "spm_en.model"), ("tgt", "spm_hi.model")):
        ckpt["meta"]["tokenizers"][side]["path"] = "ml\tokenization\\" + name
    run = tmp_path / "winrun"
    run.mkdir()
    for name in ("spm_en.model", "spm_hi.model"):
        shutil.copy(os.path.join(os.path.dirname(client.ckpt_path), name), run / name)
    torch.save(ckpt, run / "best.pt")
    model, src_tok, tgt_tok, meta = load_checkpoint(str(run / "best.pt"))
    assert src_tok.model_path.replace("\\", "/").endswith("winrun/spm_en.model")
