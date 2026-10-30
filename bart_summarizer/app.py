# =============================================================================
# app.py  -  FRONTEND (Streamlit UI only)
# All logic lives in the other modules:
#   data.py        -> XSum from HuggingFace
#   summarizer.py  -> BART generation / reconstruction / architecture info
#   noising.py     -> the paper's pre-training corruptions
#   evaluation.py  -> Lead-3 baseline, ROUGE, paper's published numbers
# Run: streamlit run app.py
# =============================================================================


# -----------------------------------------------------------------------------
# STAGE 1: IMPORTS
# -----------------------------------------------------------------------------
import random

import pandas as pd
import streamlit as st

from data import get_example, load_xsum
from evaluation import PAPER_XSUM, lead_k, novel_word_percent, rouge_corpus, rouge_single
from noising import METHODS, noise_text
from summarizer import (
    PAPER_DECODING,
    PRETRAINED_MODEL,
    XSUM_MODEL,
    get_device,
    get_model_info,
    load_model,
    reconstruct,
    summarize,
    summarize_batch,
)


# -----------------------------------------------------------------------------
# STAGE 2: PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="BART on XSum", page_icon="📝", layout="wide")


# -----------------------------------------------------------------------------
# STAGE 3: CACHED RESOURCES
# Dataset and models are expensive, so Streamlit caches them across reruns.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading XSum from HuggingFace (first run downloads ~250 MB)...")
def get_split(split: str):
    return load_xsum(split)


@st.cache_resource(show_spinner="Loading BART (first run downloads ~1.6 GB)...")
def warm_up(model_id: str):
    return load_model(model_id)


# -----------------------------------------------------------------------------
# STAGE 4: SIDEBAR - dataset split, model, decoding settings
# Defaults follow paper §5.3: beam size 5, trigram blocking (fixed in backend).
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
split = st.sidebar.selectbox("XSum split", ["test", "validation"])
model_id = st.sidebar.text_input(
    "Summarization model", XSUM_MODEL,
    help="HuggingFace id or a local folder produced by finetune.py",
)

st.sidebar.subheader("Decoding (paper §5.3)")
num_beams = st.sidebar.slider("Beam size", 1, 8, PAPER_DECODING["num_beams"])
min_len = st.sidebar.slider("Min length (tokens)", 1, 50, PAPER_DECODING["min_len"])
max_len = st.sidebar.slider("Max length (tokens)", 20, 150, PAPER_DECODING["max_len"])
length_penalty = st.sidebar.slider("Length penalty", 0.0, 3.0, PAPER_DECODING["length_penalty"], 0.1)
gen_kwargs = dict(num_beams=num_beams, min_len=min_len, max_len=max_len, length_penalty=length_penalty)

st.sidebar.caption("Device: " + ("GPU 🚀" if get_device() == "cuda" else "CPU 🐢"))


# -----------------------------------------------------------------------------
# STAGE 5: HEADER
# -----------------------------------------------------------------------------
st.title("📝 BART: Abstractive Summarization on XSum")
st.write(
    "Re-implementation of the key ideas of **Lewis et al., ACL 2020**: a denoising "
    "seq2seq pre-trained model, fine-tuned to write one-sentence summaries of BBC articles."
)

tab_sum, tab_noise, tab_eval, tab_arch = st.tabs(
    ["1 · Summarize XSum", "2 · Pre-training (denoising)", "3 · ROUGE evaluation", "4 · Architecture"]
)


# =============================================================================
# TAB 1: SUMMARIZE  (paper §3.3 sequence generation)
# Pick a real XSum article (or paste your own), generate a BART summary, and
# compare it with the human reference and the Lead-3 baseline.
# =============================================================================
with tab_sum:
    ds = get_split(split)
    n_examples = len(ds)

    # --- STAGE 5a: choose an article from the dataset ------------------------
    st.session_state.setdefault("idx", 0)
    source = st.radio("Input", ["XSum example", "My own text"], horizontal=True)

    if source == "XSum example":
        c1, c2 = st.columns([1, 3])
        if c1.button("🎲 Random example"):
            st.session_state["idx"] = random.randrange(n_examples)
        idx = c2.number_input(f"Example index (0 – {n_examples - 1})", 0, n_examples - 1, key="idx")
        ex = get_example(ds, idx)
        article, reference = ex["document"], ex["summary"]
        st.text_area("Article (first sentence already removed in XSum)", article, height=260, disabled=True)
    else:
        article = st.text_area("Paste a news article", height=260)
        reference = None

    st.session_state["article"] = article  # reused by tab 2

    # --- STAGE 5b: run BART + baseline ---------------------------------------
    if st.button("Summarize", type="primary"):
        if len(article.split()) < 20:
            st.warning("Please provide at least ~20 words.")
        else:
            warm_up(model_id)
            with st.spinner("BART is writing the summary..."):
                bart_summary = summarize(article, model_id, **gen_kwargs)
            lead3 = lead_k(article, 3)

            st.markdown("#### BART summary")
            st.success(bart_summary)
            if reference:
                st.markdown("#### Human reference (XSum)")
                st.warning(reference)
            st.markdown("#### Lead-3 baseline (first 3 sentences)")
            st.info(lead3)

            # --- STAGE 5c: metrics --------------------------------------------
            m1, m2 = st.columns(2)
            m1.metric("New words in BART summary", f"{novel_word_percent(article, bart_summary):.0f}%")
            m2.metric("New words in Lead-3", f"{novel_word_percent(article, lead3):.0f}%")

            if reference:
                rows = {
                    "BART": rouge_single(bart_summary, reference),
                    "Lead-3": rouge_single(lead3, reference),
                }
                st.caption("ROUGE F1 vs the human reference (single example, so noisy):")
                st.dataframe(pd.DataFrame(rows).T.round(1), use_container_width=True)


# =============================================================================
# TAB 2: PRE-TRAINING / DENOISING  (paper §2.2, Figure 2)
# Corrupt text with the paper's noising functions, then let the PRE-TRAINED
# BART (not fine-tuned) reconstruct the original.
# =============================================================================
with tab_noise:
    st.write(
        "BART is pre-trained by **corrupting** documents and learning to **rebuild** them. "
        "Pick a corruption below and watch the pre-trained model repair it."
    )

    default_text = " ".join(st.session_state.get("article", "").split()[:100])
    clean = st.text_area("Original text (defaults to the article from tab 1)", default_text, height=160)

    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Noising transformation", METHODS, index=5)
    seed = c2.number_input("Random seed", 0, 9999, 0)
    ratio = c3.slider("Mask ratio (infilling)", 0.1, 0.6, 0.3, 0.05)

    if clean.strip():
        noised = noise_text(clean, method, seed=int(seed), mask_ratio=ratio)
        st.markdown("**Corrupted input to the encoder**")
        st.code(noised, language=None)

        if st.button("Reconstruct with BART"):
            warm_up(PRETRAINED_MODEL)
            with st.spinner("Decoder is reconstructing the document..."):
                rebuilt = reconstruct(noised)
            st.markdown("**BART reconstruction**")
            st.success(rebuilt)
            st.caption(
                "The released BART-large was pre-trained with Text Infilling + Sentence "
                "Permutation (paper §5.1), so those settings reconstruct best."
            )


# =============================================================================
# TAB 3: ROUGE EVALUATION  (paper §5.3, Table 3)
# Run BART and Lead-3 on N real test articles and score with ROUGE, then show
# the paper's published XSum numbers beside yours.
# =============================================================================
with tab_eval:
    st.write("Evaluate on the first *N* articles of the chosen XSum split.")
    n_eval = st.slider("Number of articles", 8, 200, 32, step=8)
    batch = st.select_slider("Batch size", [1, 2, 4, 8], value=4)

    if st.button("Run evaluation"):
        ds = get_split(split)
        subset = ds.select(range(n_eval))
        docs, refs = subset["document"], subset["summary"]

        warm_up(model_id)
        preds, bar = [], st.progress(0.0, text="Generating summaries...")
        for i in range(0, n_eval, batch):
            preds += summarize_batch(docs[i:i + batch], model_id, **gen_kwargs)
            bar.progress(min((i + batch) / n_eval, 1.0), text=f"{min(i + batch, n_eval)}/{n_eval}")
        bar.empty()

        ours = {
            f"BART (yours, n={n_eval})": rouge_corpus(preds, refs),
            f"Lead-3 (yours, n={n_eval})": rouge_corpus([lead_k(d) for d in docs], refs),
        }
        table = pd.DataFrame(ours).T
        paper = pd.DataFrame(PAPER_XSUM, index=["R1", "R2", "RL"]).T
        st.dataframe(pd.concat([table, paper]).round(2), use_container_width=True)
        st.caption(
            "Small samples are noisy; the paper's numbers are on the full test set "
            "(11,334 articles)."
        )

        with st.expander("See example outputs"):
            for d, p, r in list(zip(docs, preds, refs))[:5]:
                st.markdown(f"**BART:** {p}\n\n**Reference:** {r}")
                st.divider()


# =============================================================================
# TAB 4: ARCHITECTURE  (paper §2.1)
# Reads the real model config so the numbers shown are not hard-coded.
# =============================================================================
with tab_arch:
    st.markdown(
        """
**BART = bidirectional encoder (BERT-like) + left-to-right decoder (GPT-like).**

- Standard seq2seq Transformer, but with **GeLU** activations and N(0, 0.02) init
- Each decoder layer also does **cross-attention** over the encoder's final layer
- No extra feed-forward network before word prediction (unlike BERT)
- Base: 6+6 layers · Large: 12+12 layers
- Pre-training: corrupt text → reconstruct it (text infilling + sentence permutation worked best)
- Fine-tuning for summarization: article → encoder, summary → decoder, label-smoothed cross-entropy (0.1)
        """
    )
    if st.button("Inspect loaded model"):
        warm_up(model_id)
        info = get_model_info(model_id)
        st.table(pd.DataFrame(info.items(), columns=["Property", "Value"]).set_index("Property"))


# -----------------------------------------------------------------------------
# STAGE 6: FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Lewis et al., *BART*, ACL 2020 (arXiv:1910.13461) · Dataset: XSum "
    "(EdinburghNLP/xsum) · HuggingFace Transformers + Streamlit"
)