"""Review Analysis using SimpleRNN — Streamlit frontend.

Loads the already-trained SimpleRNN model (never retrains) and classifies
IMDB-style movie reviews as Positive / Negative.

Run:
    streamlit run streamlit_app.py
"""

import io
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

MODEL_PATH = os.environ.get(
    "RNN_MODEL_PATH", "simple_rnn_imdb_optimized.h5"
)
METRICS_PATH = os.environ.get("RNN_METRICS_PATH", "test_metrics.json")
META_PATH = os.environ.get("RNN_META_PATH", "model_metadata.json")
CURVES_PATH = "loss_curves_optimized.png"
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
@st.cache_resource(show_spinner="Loading evaluation data...")
def load_eval_data():
    """Load saved test metrics + training metadata. Missing/corrupt files
    return None entries so the app keeps working with a warning."""
    metrics, meta = None, None
    for path, slot in ((METRICS_PATH, "metrics"), (META_PATH, "meta")):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:  # noqa: BLE001 - corrupt file, fall back gracefully
            continue
        if slot == "metrics":
            metrics = data
        else:
            meta = data
    return metrics, meta


def keras_summary_text(model) -> str:
    """Readable Keras model.summary() captured as text (actual live model)."""
    buf = io.StringIO()
    model.summary(print_fn=lambda line: buf.write(line + "\n"))
    return buf.getvalue()


def layer_rows(model) -> list:
    """One dict per layer: name, type, output shape, params, activation.

    All values are strings so the table serializes cleanly in Streamlit.
    """
    rows = []
    for layer in model.layers:
        cfg = layer.get_config()
        rows.append(
            {
                "Layer": str(layer.name),
                "Type": str(layer.__class__.__name__),
                "Output shape": str(layer.output_shape),
                "Params": f"{layer.count_params():,}",
                "Activation": str(cfg.get("activation", "—")),
            }
        )
    return rows


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
    metrics, meta = load_eval_data()

    tab_review, tab_summary, tab_eval = st.tabs(
        ["Review Analysis", "Model Summary", "Model Evaluation"]
    )

    # ------------------------------------------------------ Review Analysis ---
    with tab_review:
        review = st.text_area("Enter your review:", height=150, key="review")

        if st.button("Analyze Review"):
            if not review or not review.strip():
                st.warning("Please enter a review before analyzing.")
            else:
                try:
                    label, score, n_words = predict_sentiment(
                        review, word_index, model, maxlen
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    if DEBUG:
                        st.exception(exc)
                else:
                    if n_words > maxlen:
                        st.info(
                            f"Your review has {n_words} words; only the last "
                            f"{maxlen} were used (same truncation as training)."
                        )
                    st.subheader("Prediction:")
                    st.write(f"**{label}**")
                    st.write(f"Confidence: {max(score, 1 - score) * 100:.2f}%")
                    st.write(f"Raw score: {score:.4f} (threshold 0.5)")

        # Reset clears the keyed text area via on_click (which runs before the
        # script body, where widget-state edits are legal). The rerun that
        # follows every button click then wipes the previous prediction output.
        st.button("Reset", on_click=reset_analysis)

    # -------------------------------------------------------- Model Summary ---
    with tab_summary:
        st.header("Model Summary")
        rnn_layers = [l for l in model.layers if l.__class__.__name__ == "SimpleRNN"]
        units = rnn_layers[0].get_config().get("units", "?") if rnn_layers else "?"
        total = model.count_params()
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        if isinstance(model.loss, str):
            loss = model.loss
        else:
            loss = getattr(model.loss, "name", None) or getattr(
                model.loss, "__name__", "?"
            )
        try:
            lr = float(model.optimizer.learning_rate)
            lr_text = f"{lr:g}"
        except Exception:  # noqa: BLE001 - schedule or symbolic LR
            lr_text = (meta or {}).get("learning_rate", "?")
        col1, col2, col3 = st.columns(3)
        col1.metric("Recurrent type", "SimpleRNN")
        col2.metric("SimpleRNN units", units)
        col3.metric("Total params", f"{total:,}")
        col1.metric("Trainable", f"{trainable:,}")
        col2.metric("Non-trainable", f"{total - trainable:,}")
        col3.metric("Output", "sigmoid (P=positive)")
        st.subheader("Layers")
        st.dataframe(layer_rows(model), width="stretch")
        st.subheader("Training configuration")
        opt_name = model.optimizer.__class__.__name__
        train_cfg = {
            "Optimizer": opt_name,
            "Learning rate": lr_text,
            "Loss": loss,
            "Batch size": (meta or {}).get("batch_size", "?"),
            "Epochs run / max": (
                f"{(meta or {}).get('epochs_ran', '?')} / "
                f"{(meta or {}).get('max_epochs', '?')} "
                f"(best: {(meta or {}).get('best_epoch', '?')})"
            ),
            "Early stopping": str((meta or {}).get("early_stopping", "?")),
            "LR schedule": str((meta or {}).get("lr_schedule", "?")),
        }
        st.table({k: str(v) for k, v in train_cfg.items()})
        with st.expander("Keras model.summary()"):
            st.code(keras_summary_text(model))

    # ------------------------------------------------------ Model Evaluation --
    with tab_eval:
        st.header("Model Evaluation")
        if not metrics:
            st.warning(
                f"Evaluation file '{METRICS_PATH}' is missing or unreadable, "
                "so test metrics cannot be shown. The review classifier above "
                "still works."
            )
        else:
            st.write(f"Held-out IMDB test set: {metrics.get('n_test', '?')} reviews")
            m1, m2, m3 = st.columns(3)
            m1.metric("Test loss", metrics["test_loss"])
            m2.metric("Test accuracy", metrics["test_accuracy"])
            m3.metric("F1 score", metrics["f1"])
            m1.metric("Precision", metrics["precision"])
            m2.metric("Recall", metrics["recall"])
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots()
            ax.matshow(cm)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:,}", ha="center", va="center")
            ax.set_xlabel("Predicted (neg, pos)")
            ax.set_ylabel("Actual (neg, pos)")
            ax.set_title("Confusion matrix (test set)")
            st.pyplot(fig)
            dist = cm.sum(axis=1)
            fig2, ax2 = plt.subplots()
            ax2.bar(["Negative", "Positive"], dist)
            ax2.set_title("Test-set class distribution")
            st.pyplot(fig2)
        if os.path.exists(CURVES_PATH):
            st.subheader("Training curves")
            st.image(CURVES_PATH, caption="Training vs validation loss/accuracy")


if __name__ == "__main__":
    main()
