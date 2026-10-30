# =============================================================================
# noising.py  -  BART PRE-TRAINING NOISING FUNCTIONS  (paper §2.2, Figure 2)
# BART is a denoising autoencoder: corrupt a document, then learn to rebuild it.
# These are re-implementations of the 5 corruption schemes from the paper.
# For readability they work on WORDS; the real model works on BPE tokens.
# =============================================================================

import re
from collections import Counter

import numpy as np

MASK = "<mask>"  # BART's mask token string


# -----------------------------------------------------------------------------
# STAGE 1: TOKEN MASKING  (as in BERT)
# Random tokens are sampled and replaced with <mask>.
# -----------------------------------------------------------------------------
def token_masking(words, rng, p=0.15):
    return [MASK if rng.random() < p else w for w in words]


# -----------------------------------------------------------------------------
# STAGE 2: TOKEN DELETION
# Random tokens are DELETED. Unlike masking, the model must also work out
# WHICH positions are missing.
# -----------------------------------------------------------------------------
def token_deletion(words, rng, p=0.15):
    kept = [w for w in words if rng.random() >= p]
    return kept or words[:1]


# -----------------------------------------------------------------------------
# STAGE 3: TEXT INFILLING  (the paper's novel scheme, best-performing)
# Span lengths ~ Poisson(lambda=3). Each span is replaced by a SINGLE <mask>.
# 0-length spans = inserting a <mask>. Teaches the model to predict HOW MANY
# tokens are missing from a span.
# -----------------------------------------------------------------------------
def text_infilling(words, rng, mask_ratio=0.3, lam=3.0):
    n = len(words)
    budget = int(round(n * mask_ratio))       # how many words to hide in total
    covered = np.zeros(n, dtype=bool)
    spans, inserts = {}, Counter()

    for _ in range(2000):                     # safety cap on attempts
        if budget <= 0:
            break
        length = int(rng.poisson(lam))        # span length ~ Poisson(3)
        if length == 0:                       # zero-length span: just insert a mask
            inserts[int(rng.integers(0, n + 1))] += 1
            continue
        length = min(length, budget, n)
        start = int(rng.integers(0, n - length + 1))
        if covered[start:start + length].any():
            continue                          # don't overlap earlier spans
        covered[start:start + length] = True
        spans[start] = length
        budget -= length

    # rebuild the text: each span collapses into ONE mask token
    out, i = [], 0
    while i <= n:
        out.extend([MASK] * inserts.get(i, 0))
        if i == n:
            break
        if i in spans:
            out.append(MASK)
            i += spans[i]
        else:
            out.append(words[i])
            i += 1
    return out


# -----------------------------------------------------------------------------
# STAGE 4: SENTENCE PERMUTATION
# Document is split on full stops and the sentences are shuffled.
# -----------------------------------------------------------------------------
def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def sentence_permutation(text, rng):
    sents = split_sentences(text)
    order = rng.permutation(len(sents))
    return " ".join(sents[i] for i in order)


# -----------------------------------------------------------------------------
# STAGE 5: DOCUMENT ROTATION
# Pick a token uniformly at random and rotate so the document starts there.
# Trains the model to identify the start of the document.
# -----------------------------------------------------------------------------
def document_rotation(words, rng):
    k = int(rng.integers(0, len(words)))
    return words[k:] + words[:k]


# -----------------------------------------------------------------------------
# STAGE 6: DISPATCHER
# One entry point for the UI. Transformations can be composed (Figure 2); the
# large BART model used Text Infilling (30% masked) + Sentence Permutation (§5.1).
# -----------------------------------------------------------------------------
METHODS = [
    "Token Masking",
    "Token Deletion",
    "Text Infilling",
    "Sentence Permutation",
    "Document Rotation",
    "Text Infilling + Sentence Permutation (BART-large recipe)",
]


def noise_text(text, method, seed=0, p=0.15, mask_ratio=0.3, lam=3.0):
    rng = np.random.default_rng(seed)
    words = text.split()
    if not words:
        return text

    if method == "Token Masking":
        return " ".join(token_masking(words, rng, p))
    if method == "Token Deletion":
        return " ".join(token_deletion(words, rng, p))
    if method == "Text Infilling":
        return " ".join(text_infilling(words, rng, mask_ratio, lam))
    if method == "Sentence Permutation":
        return sentence_permutation(text, rng)
    if method == "Document Rotation":
        return " ".join(document_rotation(words, rng))
    if method.startswith("Text Infilling +"):
        permuted = sentence_permutation(text, rng)          # permute all sentences
        return " ".join(text_infilling(permuted.split(), rng, mask_ratio, lam))
    raise ValueError(f"Unknown method: {method}")