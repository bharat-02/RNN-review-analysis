"""Review Analysis using SimpleRNN — Streamlit frontend.

Loads the already-trained SimpleRNN model (never retrains) and classifies
IMDB-style movie reviews as Positive / Negative.

Run:
    streamlit run streamlit_app.py
"""

import os
import re

import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

MODEL_PATH = os.environ.get(
    "RNN_MODEL_PATH", "simple_rnn_imdb_optimized.h5"
)
VOCAB_SIZE = 10000
DEBUG = os.environ.get("RNN_DEBUG", "0") == "1"


# ----------------------------------------------------------------------------
# Preprocessing — identical to the training pipeline (final_train.py)
# ----------------------------------------------------------------------------
def clean_text(text: str) -> list:
    """Lowercase, strip punctuation, split into words."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def encode_review(words: list, word_index: dict, vocab_size: int = VOCAB_SIZE) -> list:
    """Map words to IMDB ids. Known word -> index+3, unknown -> 2 (OOV).

    Words outside the top-``vocab_size`` vocabulary also map to 2, exactly
    like ``imdb.load_data(num_words=...)`` does during training — otherwise
    the id would fall outside the embedding matrix and crash inference.
    """
    return [
        word_index[w] + 3
        if (w in word_index and word_index[w] + 3 < vocab_size)
        else 2
        for w in words
    ]


def preprocess_text(text: str, word_index: dict, maxlen: int):
    """Full pipeline: clean -> encode -> pad (pre-truncate/pad, like training)."""
    words = clean_text(text)
    encoded = encode_review(words, word_index)
    return sequence.pad_sequences([encoded], maxlen=maxlen), len(words)


# ----------------------------------------------------------------------------
# Artifacts — loaded once, never retrained
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading SimpleRNN model...")
def load_artifacts():
    """Load word index + trained model. Raises RuntimeError with a short message."""
    try:
        word_index = imdb.get_word_index()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the IMDB word index. Check your internet "
            "connection and restart the app."
        ) from exc
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found. Train it with "
            "`python final_train.py` or set RNN_MODEL_PATH."
        )
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load '{MODEL_PATH}'. It may be corrupted — "
            "retrain with `python final_train.py`."
        ) from exc
    maxlen = model.input_shape[1] or 300
    return word_index, model, maxlen


def predict_sentiment(text: str, word_index, model, maxlen: int):
    """Return (label, score, word_count). Raises RuntimeError on failure."""
    padded, n_words = preprocess_text(text, word_index, maxlen)
    try:
        score = float(model.predict(padded, verbose=0)[0][0])
    except Exception as exc:
        raise RuntimeError(
            "Prediction failed. The review may be in an unexpected format."
        ) from exc
    if score >= 0.65:
        label = "Positive"
    elif score <= 0.35:
        label = "Negative"
    else:
        label = "Uncertain (near decision boundary)"
    return label, score, n_words


# ----------------------------------------------------------------------------
# UI state
# ----------------------------------------------------------------------------
def reset_analysis():
    """Clear the review widget (runs before the script body, so the widget
    state can be modified legally). Streamlit reruns automatically after
    any button click, returning the app to its initial state."""
    st.session_state.review = ""


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Review Analysis using SimpleRNN")
    st.title("Review Analysis using SimpleRNN")
    st.write(
        "Enter a movie review below. A trained **SimpleRNN** classifier "
        "predicts whether it is positive or negative."
    )

    try:
        word_index, model, maxlen = load_artifacts()
    except RuntimeError as exc:
        st.error(str(exc))
        if DEBUG:
            st.exception(exc)
        st.stop()
    with st.expander("Model info"):
        st.write(f"Model file: `{MODEL_PATH}`")
        st.write(
            "Layers: "
            + ", ".join(l.__class__.__name__ for l in model.layers)
        )
        st.write(f"Sequence length: {maxlen} tokens (longer reviews are truncated)")

    review = st.text_area("Enter your review:", height=150, key="review")

    if st.button("Analyze Review"):
        if not review or not review.strip():
            st.warning("Please enter a review before analyzing.")
            return
        try:
            label, score, n_words = predict_sentiment(
                review, word_index, model, maxlen
            )
        except RuntimeError as exc:
            st.error(str(exc))
            if DEBUG:
                st.exception(exc)
            return
        if n_words > maxlen:
            st.info(
                f"Your review has {n_words} words; only the last {maxlen} "
                "were used (same truncation as training)."
            )
        st.subheader("Prediction:")
        st.write(f"**{label}**")
        st.write(f"Confidence: {max( score, 1 - score) * 100:.2f}%")
        st.write(f"Raw score: {score:.4f} (threshold 0.5)")

    # Reset clears the keyed text area via on_click (which runs before the
    # script body, where widget-state edits are legal). The rerun that follows
    # every button click then wipes the previous prediction/confidence output.
    st.button("Reset", on_click=reset_analysis)


if __name__ == "__main__":
    main()
