"""Final SimpleRNN review-analysis tests. Run: python tests/test_model.py

Covers: model loading (fresh process), preprocessing, end-to-end reviews,
prediction validity, edge cases, and no-retrain guard for the Streamlit app.
Exit code 0 = all PASS, 1 = any FAIL.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

MODEL_PATH = "simple_rnn_imdb_optimized.h5"
RESULTS = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
        RESULTS.append((name, False, str(exc)))
        print(f"{name}: FAIL ({exc})", flush=True)
    else:
        RESULTS.append((name, True, ""))
        print(f"{name}: PASS", flush=True)


# ---------------------------------------------------------------- loading ---
def t_model_loading():
    import tensorflow as tf

    assert os.path.exists(MODEL_PATH), "model file missing"
    model = tf.keras.models.load_model(MODEL_PATH)
    kinds = [l.__class__.__name__ for l in model.layers]
    assert "SimpleRNN" in kinds, f"no SimpleRNN layer: {kinds}"
    for banned in ("LSTM", "GRU", "Attention", "Transformer", "Bidirectional"):
        assert banned not in kinds, f"banned layer present: {banned}"
    assert model.input_shape[1] in (None, 300), model.input_shape
    assert model.output_shape[1] == 1, model.output_shape
    # fixed 300-token inputs (as used in training) must work
    probe = model.predict(np.zeros((1, 300), dtype=int), verbose=0)
    assert probe.shape == (1, 1) and np.all(np.isfinite(probe)), probe.shape
    for w in model.get_weights():
        assert np.all(np.isfinite(w)), "NaN/Inf in weights"
    from tensorflow.keras.datasets import imdb

    wi = imdb.get_word_index()
    assert len(wi) > 1000, "word index looks wrong"
    globals()["_model"] = model
    globals()["_wi"] = wi


# --------------------------------------------------------- preprocessing ---
def t_preprocessing():
    sys.path.insert(0, os.path.abspath("."))
    import streamlit_app as app

    assert app.clean_text("Great! Acting was AMAZING...") == [
        "great",
        "acting",
        "was",
        "amazing",
    ]
    assert app.clean_text("  ") == []
    assert app.encode_review(["great", "nosuchword_xyz"], {"great": 10}) == [13, 2]
    padded, n = app.preprocess_text("Hello WORLD!", {"hello": 5, "world": 7}, 300)
    assert padded.shape == (1, 300), padded.shape
    assert n == 2
    # exact same encoding rule as training (final_train.py data pipeline)
    assert app.encode_review(["x"], {}) == [2]


# --------------------------------------------------------------- reviews ---
REVIEWS = {
    "clearly positive": "This movie was fantastic! The acting was great and the plot was thrilling.",
    "clearly negative": "This movie was terrible, boring and a complete waste of time.",
    "ambiguous": "The acting was good but the story was boring and predictable.",
    "short": "Great movie!",
    "punctuation/case": "AMAZING!!! ...Was it though??? WORST. EVER.",
    "unknown words": "xyzblorp qwertastic zzzfnord",
    "long": " ".join(["good"] * 400),
}


def _predict(text):
    import streamlit_app as app

    model, wi = globals()["_model"], globals()["_wi"]
    label, score, n_words = app.predict_sentiment(text, wi, model, 300)
    assert isinstance(score, float), type(score)
    assert np.isfinite(score) and 0.0 <= score <= 1.0, score
    assert isinstance(label, str) and label, "empty label"
    return label, score, n_words


def t_reviews_valid():
    for name, text in REVIEWS.items():
        label, score, n = _predict(text)
        print(f"  review[{name}]: label={label} score={score:.4f} words={n}")


def t_positive_direction():
    label, score, _ = _predict(REVIEWS["clearly positive"])
    assert score > 0.5, f"positive review scored {score}"


def t_negative_direction():
    label, score, _ = _predict(REVIEWS["clearly negative"])
    assert score < 0.5, f"negative review scored {score}"


# ------------------------------------------------------------- edge cases ---
def t_edge_cases():
    import streamlit_app as app

    model, wi = globals()["_model"], globals()["_wi"]
    for text in ["", "   ", "a", " ".join(["word"] * 2000), "12345 !!!! ????"]:
        label, score, n = app.predict_sentiment(text, wi, model, 300)
        assert np.isfinite(score) and 0.0 <= score <= 1.0, (text[:20], score)
    print("  edge cases handled without exceptions")


def t_no_retrain_guard():
    src = open("streamlit_app.py").read()
    assert ".fit(" not in src, "Streamlit app must never retrain"
    assert "simple_rnn_imdb_optimized.h5" in src


if __name__ == "__main__":
    check("Model loading", t_model_loading)
    check("Preprocessing", t_preprocessing)
    check("SimpleRNN inference (7 reviews)", t_reviews_valid)
    check("Positive review test", t_positive_direction)
    check("Negative review test", t_negative_direction)
    check("Edge cases", t_edge_cases)
    check("No-retrain guard", t_no_retrain_guard)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} tests passed")
    sys.exit(1 if failed else 0)
