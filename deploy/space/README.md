---
title: TransNeural English-Hindi NMT
emoji: 🔤
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Custom Transformer translating English and Hindi
---

# TransNeural: English ↔ Hindi Neural Machine Translation

A Transformer encoder-decoder implemented from scratch in PyTorch ("Attention Is All You Need",
Vaswani et al. 2017) and trained on the IIT Bombay English-Hindi parallel corpus. No external
translation API or pre-trained model is used.

Create an account on the login screen, then:

- **Translate**: English → Hindi (and Hindi → English once that model is published), greedy or beam search.
- **History**: your own saved translations (other users cannot see them).
- **Evaluation**: training curves, dev/test BLEU and chrF++, and example outputs.

Note: this free Space has no persistent disk, so accounts and history reset when it restarts.
Model weights are downloaded at startup from the project's model repository.
API documentation: `/docs`.
