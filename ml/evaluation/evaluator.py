"""Corpus-level SacreBLEU and chrF++ evaluation with recorded metric signatures."""

from typing import Any, Dict, List

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


class TranslationEvaluator:
    """Scores hypotheses against references with sacreBLEU (BLEU) and chrF++ (word_order=2).

    The returned signatures record the exact metric configuration and sacreBLEU version, as
    required for reproducible reporting. Automatic metrics are evidence of quality, not a
    guarantee of semantic correctness.
    """

    @staticmethod
    def compute_metrics(
        hypotheses: List[str], references: List[List[str]], bleu_tokenize: str = "13a"
    ) -> Dict[str, Any]:
        """Args:
            hypotheses: one predicted translation per sample
            references: references[i] is the list of reference translations for sample i

        Returns:
            {'bleu', 'chrf2', 'bleu_signature', 'chrf2_signature', 'sacrebleu_version'}
        """
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have the same length")
        n_refs = max((len(r) for r in references), default=1)
        # sacreBLEU wants reference *streams*: stream j holds the j-th reference of every sample.
        ref_streams = [[refs[j] if j < len(refs) else "" for refs in references] for j in range(n_refs)]

        bleu = BLEU(tokenize=bleu_tokenize)
        chrf = CHRF(word_order=2)
        bleu_score = bleu.corpus_score(hypotheses, ref_streams)
        chrf_score = chrf.corpus_score(hypotheses, ref_streams)
        return {
            "bleu": round(bleu_score.score, 2),
            "chrf2": round(chrf_score.score, 2),
            "bleu_signature": str(bleu.get_signature()),
            "chrf2_signature": str(chrf.get_signature()),
            "sacrebleu_version": sacrebleu.__version__,
        }
