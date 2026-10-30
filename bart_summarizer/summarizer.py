# =============================================================================
# summarizer.py  -  BART MODEL LOGIC  (no Streamlit code in here)
# Implements:
#   - §3.3 sequence generation: fine-tuned BART as a seq2seq summarizer
#   - §5.3 decoding recipe: beam search (5 beams), trigram blocking, tuned
#          min/max length + length penalty
#   - §2.1 architecture inspection (layers, hidden size, GeLU, params)
#   - §2.2 pre-training objective: reconstruct a corrupted document
# =============================================================================

from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# -----------------------------------------------------------------------------
# STAGE 1: MODEL IDS
#   bart-large-xsum -> BART-large fine-tuned on XSum by the paper's authors (Facebook)
#   bart-large      -> the PRE-TRAINED (not fine-tuned) denoising model
# A local folder (from finetune.py) can be passed instead of an id.
# -----------------------------------------------------------------------------
XSUM_MODEL = "facebook/bart-large-xsum"
PRETRAINED_MODEL = "facebook/bart-large"
MAX_INPUT_TOKENS = 1024  # BART's encoder limit

# Decoding settings. Beam 5 + trigram blocking are from paper §5.3; the length
# limits (11-62) are the values tuned for XSum in the released checkpoint.
PAPER_DECODING = {"num_beams": 5, "min_len": 11, "max_len": 62, "length_penalty": 1.0}


# -----------------------------------------------------------------------------
# STAGE 2: DEVICE + CACHED LOADING
# Each model is loaded once per process (lru_cache). GPU is used if available.
# -----------------------------------------------------------------------------
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def load_model(model_id: str):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
    return tokenizer, model, device


# -----------------------------------------------------------------------------
# STAGE 3: SUMMARIZE (batched)   - paper §3.3 + §5.3
# Encoder reads the article; decoder WRITES the summary token by token.
# Decoding: beam search + removal of duplicated trigrams + length constraints.
# -----------------------------------------------------------------------------
def summarize_batch(
    texts,
    model_id: str = XSUM_MODEL,
    num_beams: int = 5,
    min_len: int = 11,
    max_len: int = 62,
    length_penalty: float = 1.0,
):
    tokenizer, model, device = load_model(model_id)

    # a) text -> token ids (BPE), truncated to BART's 1024 limit
    inputs = tokenizer(
        list(texts), return_tensors="pt", truncation=True,
        max_length=MAX_INPUT_TOKENS, padding=True,
    ).to(device)

    # b) autoregressive generation (inference only, so no gradients)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            num_beams=num_beams,
            min_length=min_len,
            max_length=max_len,
            length_penalty=length_penalty,
            no_repeat_ngram_size=3,   # paper: "remove duplicated trigrams in beam search"
            early_stopping=True,
        )

    # c) token ids -> text
    return [t.strip() for t in tokenizer.batch_decode(ids, skip_special_tokens=True)]


def summarize(text: str, model_id: str = XSUM_MODEL, **kwargs) -> str:
    return summarize_batch([text], model_id, **kwargs)[0]


# -----------------------------------------------------------------------------
# STAGE 4: RECONSTRUCT A CORRUPTED DOCUMENT   - paper §2.2
# The pre-trained BART (facebook/bart-large) was trained to map a noised
# document back to the original. Here we feed it text containing <mask> tokens
# and let the decoder regenerate the full, uncorrupted document.
# -----------------------------------------------------------------------------
def reconstruct(noised_text: str, model_id: str = PRETRAINED_MODEL, num_beams: int = 4) -> str:
    tokenizer, model, device = load_model(model_id)
    inputs = tokenizer(
        noised_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
    ).to(device)

    max_len = min(inputs["input_ids"].shape[1] * 2 + 10, MAX_INPUT_TOKENS)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            num_beams=num_beams,
            min_length=0,
            max_length=max_len,
            no_repeat_ngram_size=0,   # reconstruction may legitimately repeat phrases
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


# -----------------------------------------------------------------------------
# STAGE 5: ARCHITECTURE INFO   - paper §2.1
# Reads the loaded model's config so the UI can show the paper's numbers
# (large = 12 enc + 12 dec layers, d_model 1024, GeLU activations).
# -----------------------------------------------------------------------------
def get_model_info(model_id: str = XSUM_MODEL) -> dict:
    _, model, _ = load_model(model_id)
    c = model.config
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {
        "Encoder layers (bidirectional)": str(c.encoder_layers),
        "Decoder layers (autoregressive)": str(c.decoder_layers),
        "Hidden size (d_model)": str(c.d_model),
        "Attention heads": str(c.encoder_attention_heads),
        "Feed-forward size": str(c.encoder_ffn_dim),
        "Activation": str(c.activation_function),
        "Vocabulary (byte-level BPE)": str(c.vocab_size),
        "Parameters": f"{params:.0f}M",
    }