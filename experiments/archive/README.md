# Archive: first training attempt (before the pipeline fixes)

- `train_colab_v0.ipynb` — the original Colab run. It no longer runs with the current scripts
  (their options changed); use `experiments/train_kaggle.ipynb` instead.
- `baseline_experiment_v0.json` — its result: **BLEU 0.0 / chrF++ 0.0**, empty outputs.

Why it failed (diagnosed from the notebook output):
1. Only 5,000 training pairs were used (`max_train_samples: 5000` in the config silently
   capped the data the notebook asked for).
2. No learning-rate warmup (required for stable Transformer training, Vaswani et al. 2017).
3. Validation loss *rose* (9.47 → 9.82) while training loss fell: overfitting/memorisation.

After the fixes (full 1.37M-pair corpus, warmup + inverse-sqrt schedule, dev-BLEU model
selection, length-bucketed batching) the small model reached dev BLEU 12.9 after 5 epochs.
Kept as the "before" evidence for the baseline → final comparison.
