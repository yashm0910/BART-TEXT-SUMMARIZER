# =============================================================================
# data.py  -  DATASET LAYER
# Loads the real XSum dataset (Narayan et al., 2018) from HuggingFace.
# Each example has:  document (news article), summary (ONE-sentence summary), id
# Splits: train (204k) / validation (11k) / test (11k)
#
# NOTE (paper §6): XSum's `document` already has the first sentence REMOVED
# (that sentence is the reference summary), so there is no easy extractive
# summary - that is why XSum is the "highly abstractive" benchmark.
# =============================================================================

from datasets import load_dataset

DATASET_ID = "EdinburghNLP/xsum"


# -----------------------------------------------------------------------------
# STAGE 1: LOAD A SPLIT
# Equivalent to:  ds = load_dataset("EdinburghNLP/xsum")
# Different `datasets` versions handle this repo differently (script-based vs
# parquet), so we try 3 strategies in order and use the first that works.
# -----------------------------------------------------------------------------
def load_xsum(split: str = "test"):
    strategies = [
        {},                                        # 1) plain call
        {"trust_remote_code": True},               # 2) older script-based loading
        {"revision": "refs/convert/parquet"},      # 3) auto-converted parquet files
    ]
    errors = []
    for kwargs in strategies:
        try:
            return load_dataset(DATASET_ID, split=split, **kwargs)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{kwargs or 'default'}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Could not load XSum from HuggingFace. Tried:\n  " + "\n  ".join(errors)
    )


# -----------------------------------------------------------------------------
# STAGE 2: SINGLE-EXAMPLE ACCESSOR
# Returns a plain dict so the frontend never depends on the datasets library.
# -----------------------------------------------------------------------------
def get_example(ds, idx: int) -> dict:
    row = ds[int(idx)]
    return {"id": row["id"], "document": row["document"], "summary": row["summary"]}