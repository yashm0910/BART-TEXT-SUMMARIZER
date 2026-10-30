# =============================================================================
# evaluation.py  -  BASELINE + METRICS  (paper §5.3, Table 3)
#   - Lead-3 baseline: first 3 sentences of the article
#   - ROUGE-1 / ROUGE-2 / ROUGE-L: the metrics the paper reports
#   - Paper's published XSum numbers, so your results can be compared
# =============================================================================

import re

from rouge_score import rouge_scorer

from noising import split_sentences


# -----------------------------------------------------------------------------
# STAGE 1: LEAD-3 BASELINE (extractive)
# The paper's simple baseline. It does badly on XSum (R1 16.30) because XSum
# summaries are abstractive - copying leading sentences doesn't work.
# -----------------------------------------------------------------------------
def lead_k(document: str, k: int = 3) -> str:
    return " ".join(split_sentences(document)[:k])


# -----------------------------------------------------------------------------
# STAGE 2: ROUGE
# F1 for ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common
# subsequence), with stemming, averaged over examples and scaled to 0-100.
# -----------------------------------------------------------------------------
_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge_single(prediction: str, reference: str) -> dict:
    s = _scorer.score(reference, prediction)
    return {"R1": 100 * s["rouge1"].fmeasure,
            "R2": 100 * s["rouge2"].fmeasure,
            "RL": 100 * s["rougeL"].fmeasure}


def rouge_corpus(predictions, references) -> dict:
    rows = [rouge_single(p, r) for p, r in zip(predictions, references)]
    n = max(len(rows), 1)
    return {k: sum(r[k] for r in rows) / n for k in ("R1", "R2", "RL")}


# -----------------------------------------------------------------------------
# STAGE 3: "HOW ABSTRACTIVE?" METRIC
# % of summary words that never occur in the source. High = paraphrasing.
# -----------------------------------------------------------------------------
def novel_word_percent(source: str, summary: str) -> float:
    tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())
    src, words = set(tok(source)), tok(summary)
    if not words:
        return 0.0
    return 100 * sum(w not in src for w in words) / len(words)


# -----------------------------------------------------------------------------
# STAGE 4: PAPER'S REPORTED RESULTS  (Table 3, XSum columns: R1, R2, RL)
# Used by the UI to put your measured scores next to published ones.
# -----------------------------------------------------------------------------
PAPER_XSUM = {
    "Lead-3 (paper)": (16.30, 1.60, 11.95),
    "PTGEN (paper)": (29.70, 9.21, 23.24),
    "BERTSUMABS (paper)": (38.76, 16.33, 31.15),
    "BERTSUMEXTABS (paper)": (38.81, 16.50, 31.27),
    "BART (paper)": (45.14, 22.27, 37.25),
}