# 🎬 RNN Review Analysis

### IMDB Movie Review Sentiment Analysis with Recurrent Neural Networks

An end-to-end Natural Language Processing (NLP) project that uses a **Simple Recurrent Neural Network (RNN)** to classify movie reviews based on sentiment. The project covers the complete workflow from text preprocessing and word encoding to sequence padding, embedding, model inference, and deployment through an interactive **Streamlit** application.

The trained TensorFlow/Keras model is integrated into a lightweight web application where users can enter a movie review and receive a sentiment prediction together with the model's output score.

---

## 🚀 Live Demo

**Try the deployed application:**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://rnn-review-analysis-oaaljqtwwe3dymse8tkez8.streamlit.app/)

👉 **[Open RNN Review Analysis](https://rnn-review-analysis-oaaljqtwwe3dymse8tkez8.streamlit.app/)**

---

## 📌 Project Overview

Sentiment analysis is a fundamental NLP task used to determine the emotional orientation of text. In this project, an RNN is trained on the **IMDB Movie Review Dataset** to learn patterns associated with positive and negative movie reviews.

The trained model is then exposed through a Streamlit interface, making it possible to perform real-time inference on user-provided reviews.

### End-to-End Pipeline

```text
                 ┌──────────────────────┐
                 │    Movie Review      │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Text Preprocessing   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Word Index Encoding  │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Sequence Padding     │
                 │      maxlen=500      │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │   Word Embedding     │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │     Simple RNN       │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Sigmoid Probability  │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Sentiment Prediction │
                 └──────────────────────┘
```

---

## ✨ Key Features

- 🧠 **Simple RNN-based sentiment classification**
- 🎬 **IMDB movie review dataset**
- 🔤 Text-to-integer word encoding
- 📏 Fixed-length sequence padding
- 📚 Word embedding for dense text representation
- 🤖 TensorFlow/Keras trained model
- 📊 Prediction score exposed in the application
- 🌐 Interactive Streamlit interface
- 🔄 Reset functionality for repeated predictions
- 🚀 Public deployment through Streamlit

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python** | Core programming language |
| **TensorFlow 2.15** | Deep learning framework |
| **Keras** | Neural network/model API |
| **Simple RNN** | Sequential text classification |
| **IMDB Dataset** | Sentiment-analysis dataset |
| **NumPy** | Numerical computation |
| **Scikit-learn** | ML utilities and evaluation support |
| **Streamlit** | Interactive web application |
| **Jupyter Notebook** | Experimentation and model development |
| **TensorBoard** | Training visualization/tooling |

---

## 📂 Repository Structure

```text
RNN-review-analysis/
│
├── app.py
│   └── Streamlit inference application
│
├── simplernn.ipynb
│   └── RNN model development and training workflow
│
├── embedding.ipynb
│   └── Word embedding experiments
│
├── prediction.ipynb
│   └── Model loading and prediction experiments
│
├── simple_rnn_imdb.h5
│   └── Trained TensorFlow/Keras RNN model
│
├── requirements.txt
│   └── Python project dependencies
│
└── README.md
    └── Project documentation
```

---

# 🧠 Model Workflow

## 1. Dataset

The project uses the **IMDB Movie Review Dataset** available through TensorFlow/Keras.

The original task is binary sentiment classification:

```text
0 → Negative
1 → Positive
```

The model learns from labeled movie reviews and predicts the sentiment of previously unseen text.

---

## 2. Text Preprocessing

Neural networks cannot directly consume raw text. Each review is therefore transformed into a numerical sequence using the IMDB vocabulary.

For example:

```text
"This movie was amazing"
```

is converted into a sequence of integer word IDs.

Unknown words are mapped to the configured out-of-vocabulary representation.

---

## 3. Sequence Padding

Movie reviews can contain different numbers of words. To provide the model with a consistent input shape, each sequence is padded/truncated to a maximum length of **500 tokens**.

```python
sequence.pad_sequences(
    [encoded_review],
    maxlen=500
)
```

This produces a fixed-size input suitable for the neural network.

---

## 4. Word Embedding

The integer-encoded tokens are transformed into dense vector representations through an embedding layer.

Conceptually:

```text
Token ID
   ↓
Embedding Layer
   ↓
Dense Vector
   ↓
Semantic Representation
```

The embedding allows the network to learn useful numerical representations of words during training.

---

## 5. Recurrent Neural Network

The embedded sequence is processed by a **Simple RNN**.

Unlike a model that treats every token independently, an RNN processes sequential information while maintaining a hidden state. This makes it suitable for learning patterns where word order contributes to meaning.

```text
Word 1 → Word 2 → Word 3 → ... → Word N
   ↓        ↓        ↓              ↓
 ┌────────────────────────────────────┐
 │            Simple RNN              │
 └──────────────────┬─────────────────┘
                    ↓
              Classification
                    ↓
             Sigmoid Output
```

---

## 6. Sentiment Prediction

The final classifier produces a sigmoid score between `0` and `1`.

The Streamlit application interprets the score using the following application-level thresholds:

| Prediction Score | Application Output |
|---:|---|
| `<= 0.35` | 🔴 Negative |
| `0.35 < score < 0.65` | 🟡 Neutral |
| `>= 0.65` | 🟢 Positive |

> **Important:** The underlying RNN is trained as a **binary classifier** (positive vs. negative). The `Neutral` label is an application-level interpretation of predictions close to the decision boundary; it is **not a separately trained third class**.

---

# 🌐 Streamlit Application

The application is implemented in `app.py`.

It:

1. Loads the trained `simple_rnn_imdb.h5` model.
2. Loads the IMDB word index.
3. Converts user-entered text into integer sequences.
4. Pads the sequence to 500 tokens.
5. Runs inference with the trained RNN.
6. Converts the output score into a sentiment label.
7. Displays the sentiment and prediction score.

### Application Controls

- **Movie Review** — input area for a new review
- **Classify** — runs model inference
- **Prediction Score** — displays the model output
- **Reset** — clears the current review

---

# 🧪 Example Predictions

### Positive Review

```text
This movie was absolutely fantastic. The acting was excellent,
the story was engaging, and I loved every minute of it.
```

Expected application output:

```text
Sentiment: Positive
```

### Negative Review

```text
This movie was extremely disappointing. The story was boring,
the acting was weak, and the ending made no sense.
```

Expected application output:

```text
Sentiment: Negative
```

### Mixed / Uncertain Review

```text
The acting was excellent and the movie looked beautiful,
but the story was boring and predictable.
```

A score around the decision boundary may be displayed as:

```text
Sentiment: Neutral
Prediction Score: 0.6000
```

---

# ⚙️ Installation & Setup

## Prerequisites

Make sure you have:

- Python 3.x
- Git
- pip
- A virtual environment (recommended)

---

## 1. Clone the Repository

```bash
git clone https://github.com/bharat-02/RNN-review-analysis.git
```

## 2. Navigate to the Project

```bash
cd RNN-review-analysis
```

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

### macOS / Linux

```bash
python3 -m venv venv
```

## 4. Activate the Environment

### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser, enter a movie review, and click **Classify**.

---

# 📓 Notebooks

### `simplernn.ipynb`

Contains the core model-development workflow, including:

- Dataset loading
- Data preparation
- RNN architecture
- Model training
- Model evaluation
- Model persistence

### `embedding.ipynb`

Contains experiments related to:

- Token representation
- Word embeddings
- Embedding layers
- Text representation

### `prediction.ipynb`

Contains experiments for:

- Loading the trained model
- Preparing new reviews
- Running inference
- Testing predictions

---

# 📦 Trained Model

The trained model is included in the repository as:

```text
simple_rnn_imdb.h5
```

The Streamlit application loads it with TensorFlow/Keras:

```python
from tensorflow.keras.models import load_model

model = load_model("simple_rnn_imdb.h5")
```

This allows the deployed application to perform inference without retraining the model at startup.

---

# 🔍 Implementation Details

The inference pipeline in `app.py` follows the same vocabulary and sequence representation used by the IMDB dataset.

The application:

```python
word_index = imdb.get_word_index()
```

creates the IMDB vocabulary, then maps user-entered words to integer IDs before padding:

```python
padded_review = sequence.pad_sequences(
    [encoded_review],
    maxlen=500
)
```

The trained model then generates the prediction:

```python
prediction = model.predict(preprocessed_input)
```

The resulting score is interpreted into an application-level sentiment category.

---

# 🚀 Deployment

The application is deployed using **Streamlit**.

### Live Application

👉 **[https://rnn-review-analysis-oaaljqtwwe3dymse8tkez8.streamlit.app/](https://rnn-review-analysis-oaaljqtwwe3dymse8tkez8.streamlit.app/)**

The deployment uses the repository's Streamlit entry point:

```text
app.py
```

and installs the dependencies listed in:

```text
requirements.txt
```

---

# 📈 Future Improvements

The current implementation focuses on demonstrating an end-to-end RNN sentiment-analysis workflow. Potential improvements include:

- [ ] Replace Simple RNN with **LSTM**
- [ ] Experiment with **GRU**
- [ ] Add dropout and regularization
- [ ] Perform systematic hyperparameter tuning
- [ ] Add confusion matrix and classification report
- [ ] Add training/validation metric visualizations
- [ ] Integrate TensorBoard experiment tracking
- [ ] Add confidence/probability visualization
- [ ] Support batch prediction for multiple reviews
- [ ] Add model versioning
- [ ] Improve text normalization and preprocessing
- [ ] Train a dedicated three-class sentiment model
- [ ] Containerize the application with Docker
- [ ] Add automated testing and CI/CD

---

# 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Natural Language Processing
- Text preprocessing
- Tokenization and vocabulary mapping
- Sequence padding
- Word embeddings
- Recurrent Neural Networks
- Binary sentiment classification
- Sigmoid-based probability output
- TensorFlow/Keras
- Model training and inference
- Streamlit application development
- ML model deployment

---

# 💼 Why This Project Matters

This project demonstrates more than model training alone. It connects the major components of a practical machine-learning workflow:

```text
Data
 ↓
Preprocessing
 ↓
Feature Representation
 ↓
Model Development
 ↓
Training
 ↓
Inference
 ↓
Application Integration
 ↓
Deployment
```

This makes the repository a compact example of taking an NLP/deep-learning model from experimentation to a usable application.

---



# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

Your feedback, suggestions, and contributions are welcome.

---

## 📄 License

This project is developed for educational and learning purposes.
