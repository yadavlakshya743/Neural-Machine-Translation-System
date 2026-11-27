"""Tests for the training/inference pipeline: masking under fp16, LR schedule, batching,
batched decoding, learning on a toy task, checkpoint resume and tokenizer mismatch."""

import math
import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ml.data.dataset import EncodedCorpus, EncodedPairDataset, TokenBucketBatchSampler, pad_collate_fn
from ml.decoding.strategy import BeamSearchDecoder, GreedyDecoder
from ml.evaluation.evaluator import TranslationEvaluator
from ml.tokenization.tokenizer import SubwordTokenizer
from ml.training.trainer import TrainConfig, Trainer, inverse_sqrt_lr
from ml.transformer.attention import ScaledDotProductAttention
from ml.transformer.model import TransformerSeq2Seq

PAD, SOS, EOS = SubwordTokenizer.PAD_ID, SubwordTokenizer.SOS_ID, SubwordTokenizer.EOS_ID


def tiny_model(vocab=24, d=32, layers=2, dropout=0.0, tie=False):
    torch.manual_seed(0)
    return TransformerSeq2Seq(vocab, vocab, d_model=d, n_heads=4, n_encoder_layers=layers,
                              n_decoder_layers=layers, d_ff=64, dropout=dropout, tie_embeddings=tie)


# --------------------------------------------------------------------------- model
def test_masking_does_not_overflow_in_float16():
    attn = ScaledDotProductAttention(dropout=0.0)
    q = torch.randn(1, 1, 3, 8, dtype=torch.float16)
    mask = torch.tensor([[[[True, True, False]]]])
    _, w = attn(q.float(), q.float(), q.float(), mask)
    scores = torch.randn(1, 1, 3, 3, dtype=torch.float16)
    # Old code used -1e9, which raises for float16; finfo.min must work.
    scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    assert torch.allclose(w[..., 2], torch.zeros(1, 1, 3))


def test_tied_embeddings_share_weights_and_config_roundtrip():
    m = tiny_model(tie=True)
    assert m.generator.weight is m.decoder.embedding.token_embedding.weight
    m2 = TransformerSeq2Seq.from_config(m.config)
    m2.load_state_dict(m.state_dict())
    assert m2.config == m.config


# ------------------------------------------------------------------- optimization
def test_inverse_sqrt_schedule_warms_up_then_decays():
    assert inverse_sqrt_lr(1, 100) == pytest.approx(0.01)
    assert inverse_sqrt_lr(100, 100) == pytest.approx(1.0)
    assert inverse_sqrt_lr(400, 100) == pytest.approx(0.5)
    assert inverse_sqrt_lr(50, 100) < inverse_sqrt_lr(100, 100) > inverse_sqrt_lr(200, 100)


# ------------------------------------------------------------------------ batching
def test_bucket_sampler_covers_every_index_once_and_respects_budget():
    rng = np.random.default_rng(0)
    src = rng.integers(3, 60, size=500)
    tgt = rng.integers(3, 60, size=500)
    s = TokenBucketBatchSampler(src, tgt, max_tokens=300, seed=1, pool_factor=4)
    s.set_epoch(1)
    batches = list(s)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(500))
    for b in batches:
        assert max(max(src[i], tgt[i]) for i in b) * len(b) <= 300

    # Deterministic per epoch, different across epochs, and resumable mid-epoch.
    s.set_epoch(1)
    assert list(s) == batches
    s.set_epoch(2)
    assert list(s) != batches
    s.set_epoch(1, start_batch=5)
    assert list(s) == batches[5:]


def test_encoded_corpus_drops_overlong_and_adds_specials(tmp_path):
    class FakeTok:
        def encode_batch(self, texts):
            return [[4 + (ord(c) % 10) for c in t] for t in texts]

    corpus, dropped = EncodedCorpus.from_texts(["ab", "abcdefgh", "c"], ["xy", "z", "w"],
                                               FakeTok(), FakeTok(), max_len=6)
    assert len(corpus) == 2 and dropped == 1
    path = str(tmp_path / "c.npz")
    corpus.save(path)
    loaded = EncodedCorpus.load(path)
    src, tgt = EncodedPairDataset(loaded)[0]
    assert src[0].item() == SOS and src[-1].item() == EOS and len(src) == 4


# ------------------------------------------------------------------------ decoding
def _padded(seqs):
    longest = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (longest - len(s)) for s in seqs])


def test_batched_greedy_matches_one_by_one():
    model = tiny_model().double().eval()
    dec = GreedyDecoder(model, SOS, EOS, PAD)
    sents = [[SOS, 5, 6, 7, EOS], [SOS, 9, EOS], [SOS, 4, 8, 10, 11, 12, EOS]]
    batched = dec.decode(_padded(sents), max_len=12)
    single = [dec.decode(torch.tensor([s]), max_len=12)[0] for s in sents]
    assert batched == single


def test_beam_size_one_equals_greedy_and_batched_beam_matches_single():
    model = tiny_model().double().eval()
    greedy = GreedyDecoder(model, SOS, EOS, PAD)
    beam = BeamSearchDecoder(model, SOS, EOS, PAD, beam_size=3, alpha=0.6)
    sents = [[SOS, 5, 6, 7, EOS], [SOS, 9, EOS], [SOS, 4, 8, 10, 11, 12, EOS]]
    src = _padded(sents)
    assert beam.decode(src, max_len=10, beam_size=1, alpha=0.0) == greedy.decode(src, max_len=10)
    batched = beam.decode(src, max_len=10)
    single = [beam.decode(torch.tensor([s]), max_len=10)[0] for s in sents]
    assert batched == single
    assert all(len(x) <= 10 for x in batched)


# ------------------------------------------------- learning + checkpoint + resume
def _copy_task_loader(n=256, vocab=24, seed=0):
    rng = np.random.default_rng(seed)
    seqs = [list(rng.integers(4, vocab, size=rng.integers(3, 8))) for _ in range(n)]
    ids = np.concatenate([np.array(s, dtype=np.int32) for s in seqs])
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(s) for s in seqs], out=off[1:])
    corpus = EncodedCorpus(ids, off, ids.copy(), off.copy())
    sampler = TokenBucketBatchSampler(corpus.src_lengths, corpus.tgt_lengths, max_tokens=320, seed=seed)
    return DataLoader(EncodedPairDataset(corpus), batch_sampler=sampler, collate_fn=pad_collate_fn), seqs


def test_model_learns_copy_task_and_resumes(tmp_path):
    torch.manual_seed(0)
    loader, seqs = _copy_task_loader()
    model = tiny_model(d=64, layers=2, dropout=0.0)
    cfg = TrainConfig(learning_rate=3e-3, warmup_steps=50, label_smoothing=0.0, epochs=25,
                      patience=0, select_metric="loss", save_every_steps=10_000, log_interval=10_000,
                      use_amp=False)
    meta = {"tokenizers": {"src": {"sha256": "aaa"}, "tgt": {"sha256": "bbb"}}}
    trainer = Trainer(model, loader, loader, config=cfg, device="cpu", run_dir=str(tmp_path), meta=meta)
    first_loss = trainer.validate()
    assert trainer.fit() == "COMPLETED"
    final_loss = trainer.validate()
    assert final_loss < 0.35 * first_loss, (first_loss, final_loss)

    # The trained model should actually copy most sequences with greedy decoding.
    dec = GreedyDecoder(model, SOS, EOS, PAD)
    sample = seqs[:40]
    out = dec.decode(_padded([[SOS] + list(map(int, s)) + [EOS] for s in sample]), max_len=12)
    exact = sum(o == list(map(int, s)) for o, s in zip(out, sample)) / len(sample)
    assert exact > 0.5, exact

    assert os.path.exists(trainer.best_path) and os.path.exists(trainer.metrics_path)

    # Resume restores optimizer step and state.
    t2 = Trainer(tiny_model(d=64, layers=2), loader, loader, config=cfg, device="cpu",
                 run_dir=str(tmp_path), meta=meta)
    t2.resume(trainer.last_path)
    assert t2.state.global_step == trainer.state.global_step
    assert t2.state.epoch == cfg.epochs + 1

    # Resuming with a different tokenizer must be refused.
    bad = {"tokenizers": {"src": {"sha256": "zzz"}, "tgt": {"sha256": "bbb"}}}
    t3 = Trainer(tiny_model(d=64, layers=2), loader, loader, config=cfg, device="cpu",
                 run_dir=str(tmp_path), meta=bad)
    with pytest.raises(RuntimeError, match="tokenizer differs"):
        t3.resume(trainer.last_path)


def test_time_budget_stops_and_saves(tmp_path):
    loader, _ = _copy_task_loader(n=64)
    cfg = TrainConfig(epochs=50, max_hours=0.0, use_amp=False, warmup_steps=10, log_interval=10_000)
    trainer = Trainer(tiny_model(), loader, loader, config=cfg, device="cpu", run_dir=str(tmp_path))
    assert trainer.fit() == "TIME_BUDGET_REACHED"
    assert os.path.exists(trainer.last_path)
    assert trainer.state.global_step == 1 and trainer.state.batch_in_epoch == 1


# ---------------------------------------------------------------------- evaluation
def test_evaluator_reports_scores_and_signatures():
    hyps = ["the cat sat on the mat", "hello world"]
    refs = [["the cat sat on the mat"], ["hello world"]]
    r = TranslationEvaluator.compute_metrics(hyps, refs)
    assert r["bleu"] == pytest.approx(100.0) and r["chrf2"] == pytest.approx(100.0)
    assert "nrefs:1" in r["bleu_signature"] and "nw:2" in r["chrf2_signature"]
