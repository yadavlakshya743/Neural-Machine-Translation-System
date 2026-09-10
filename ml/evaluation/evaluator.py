"""Evaluation module for computing SacreBLEU and chrF++ benchmarks."""

from typing import List, Dict, Any


class TranslationEvaluator:
    """Evaluates generated translation hypotheses against ground truth references using BLEU and chrF++."""

    @staticmethod
    def compute_metrics(
        hypotheses: List[str], references: List[List[str]]
    ) -> Dict[str, float]:
        """Args:
            hypotheses: List of predicted translation strings
            references: List of reference translation lists (each sample can have 1 or more refs)

        Returns:
            Dict containing 'bleu' and 'chrf2' scores.
        """
        results = {}

        # 1. SacreBLEU computation
        try:
            import sacrebleu

            # SacreBLEU expects list of references where references[i] is a list of refs for the i-th hypothesis,
            # or list of reference streams.
            ref_streams = list(zip(*references)) if references and isinstance(references[0], list) else [references]
            bleu = sacrebleu.corpus_bleu(hypotheses, ref_streams)
            results["bleu"] = round(bleu.score, 2)

            chrf = sacrebleu.corpus_chrf(hypotheses, ref_streams, word_order=2)
            results["chrf2"] = round(chrf.score, 2)

        except ImportError:
            # Basic fallback calculation if sacrebleu is not installed in local environment yet
            results["bleu"] = 0.0
            results["chrf2"] = 0.0
            results["note"] = "Install sacrebleu for exact corpus metric computation."

        return results
