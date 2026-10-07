# Review Analysis using SimpleRNN

IMDB movie-review sentiment analysis with a **SimpleRNN-based** deep learning
model and a Streamlit frontend. The recurrent core is **SimpleRNN only** —
no LSTM, GRU, Bidirectional RNN, Transformer, or Attention layers are used
anywhere in the model or the app.

## Features

- Review text input (Streamlit text area)
- Text preprocessing (lowercasing, punctuation stripping)
- Tokenization via the IMDB word index (OOV handling)
- Sequence generation + fixed-length padding
- SimpleRNN-based sentiment prediction with confidence score
- Test-set evaluation script (accuracy, precision, recall, F1, confusion matrix)
- Streamlit web interface (`streamlit_app.py`)

## Tech Stack

- Python
- TensorFlow / Keras (model training and inference)
- SimpleRNN (sequence classifier)
- NumPy (numerical computation)
- scikit-learn (evaluation metrics)
- Streamlit (web application)
- Matplotlib (training-curve plots)

## Model Architecture

Final optimized model (`simple_rnn_imdb_optimized.h5`, 1,321,217 parameters):

```text
Input (300 tokens)
 ↓
Embedding (vocab 10000, dim 128)
 ↓
SimpleRNN (128 units, tanh, dropout 0.2)
 ↓
Dropout (0.3)
 ↓
Dense (64 units, relu)
 ↓
Dropout (0.3)
 ↓
Dense (1 unit, sigmoid)
```

## Dataset

- Source: IMDB movie-review dataset via `tensorflow.keras.datasets.imdb`
  (`num_words=10000`)
- 25,000 training + 25,000 test reviews, balanced (50% positive / 50% negative)
- Mean review length ~239 words (train) / ~231 words (test)
- Split used here: 20,000 train / 5,000 validation / 25,000 test

## Preprocessing

```text
Raw Review
 ↓
Lowercase + punctuation removal
 ↓
Word → ID mapping (known word: index+3, unknown: 2 = OOV)
 ↓
Pre-padding / pre-truncation to 300 tokens
 ↓
SimpleRNN
 ↓
Sigmoid score → sentiment label
```

The Streamlit app reuses exactly this pipeline (`clean_text`, `encode_review`,
`preprocess_text` in `streamlit_app.py`). No separate scaler or tokenizer file
is needed — the IMDB word index is loaded from the Keras dataset.

## Training

- Optimizer: Adam, learning rate 1e-3, `clipnorm=1.0`
- Loss: `binary_crossentropy`
- Batch size: 64, up to 15 epochs (best checkpoint: epoch 7)
- Early stopping: `patience=5`, `restore_best_weights=True`
- Learning-rate scheduling: `ReduceLROnPlateau(factor=0.5, patience=2)`
- Regularization: dropout (0.2 in RNN, 0.3/0.3 after RNN + dense)
- Command: `python final_train.py`

## Model Performance

Measured on the 25,000-review IMDB test set:

| Metric | Before (baseline) | After (optimized) |
|---|---:|---:|
| Test Loss | 0.4636 | 0.4175 |
| Accuracy | 0.7989 | 0.8282 |
| Precision | 0.8098 | 0.8224 |
| Recall | 0.7813 | 0.8373 |
| F1 Score | 0.7953 | 0.8298 |
| Validation Accuracy | 0.8112 | 0.8294 |

Baseline was `Embedding → SimpleRNN(128, relu) → Dense(1)` at maxlen 500 with
no regularization (its epoch-1 loss exploded to ~2.3e11, motivating the switch
to `tanh` + gradient clipping). See `loss_curves_optimized.png`.

## Hardware

```text
CPU: Intel Core i5-13450HX
RAM: 24 GB
GPU: NVIDIA RTX 3050 6 GB
```

Note: TensorFlow 2.15 on native Windows is CPU-only, so training ran on the
CPU (~15 s/epoch, ~3 min total). GPU VRAM usage: 0 MB.

## Installation

```bash
git clone https://github.com/bharat-02/RNN-review-analysis.git
cd RNN-review-analysis

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Evaluation

```bash
python evaluate.py
```

## Prediction (CLI)

```bash
python predict.py "This movie was fantastic! The acting was great."
```

## Run Streamlit

```bash
streamlit run streamlit_app.py
```

Then open the printed local URL (typically `http://localhost:8501`), enter a
review, and click **Analyze Review**. The app loads the already-trained model —
it never retrains. Set `RNN_DEBUG=1` to show full error traces.

## Project Structure

```text
RNN-review-analysis/
│
├── streamlit_app.py            # Streamlit frontend (recommended entry point)
├── app.py / app.1.py           # Streamlit app copies (same optimized pipeline)
├── final_train.py              # Optimized SimpleRNN training script
├── evaluate.py                 # Test-set evaluation
├── predict.py                  # Single-review CLI inference
├── simple_rnn_imdb_optimized.h5# Final SimpleRNN model (maxlen 300)
├── simple_rnn_imdb.h5          # Original baseline model (reference)
├── loss_curves_optimized.png   # Training/validation curves
├── history_optimized.json      # Training history
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── .gitignore
```

## Future Improvements

- Tune SimpleRNN width/depth and sequence length around the current setup
- Tune dropout/weight decay and LR schedules for the same SimpleRNN core
- Add batch prediction for multiple reviews
- Add probability calibration and confidence visualization
- Improve text normalization (e.g. contraction handling)
- Add automated tests and CI
