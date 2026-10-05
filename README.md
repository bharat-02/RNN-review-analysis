# 🎬 RNN Review Analysis

### IMDB Movie Review Sentiment Analysis using Recurrent Neural Networks

An end-to-end **Natural Language Processing (NLP)** project that uses a **Recurrent Neural Network (RNN)** to analyze movie reviews and predict their sentiment.

The project includes model development, text preprocessing, prediction, and a deployed **Streamlit web application** for real-time sentiment analysis.

---

## 🚀 Live Demo

### 👉 [Try the Live Demo](https://rnn-review-analysis-oaaljqtwwe3dymse8tkez8.streamlit.app/)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge\&logo=streamlit)](https://rnn-review-analysis-oaaljqtwwe3dymse8tkez8.streamlit.app/)

Enter a movie review and get an instant sentiment prediction from the trained RNN model.

---

## 📌 Overview

Sentiment analysis is a common NLP task used to determine the emotional tone of textual data.

In this project, an **RNN-based deep learning model** is trained on the **IMDB Movie Review Dataset** to learn patterns associated with positive and negative movie reviews.

The trained model is then integrated into a **Streamlit application**, allowing users to enter their own movie reviews and receive predictions through an interactive web interface.

### The project covers the complete workflow:

```text
Movie Review
     ↓
Text Preprocessing
     ↓
Word Encoding
     ↓
Sequence Padding
     ↓
Word Embedding
     ↓
Recurrent Neural Network
     ↓
Sigmoid Output
     ↓
Sentiment Prediction
```

---

## ✨ Features

* 🧠 RNN-based sentiment classification
* 🎬 IMDB movie review dataset
* 🔤 Text preprocessing and word encoding
* 📚 Word embedding
* 🔢 Sequence padding
* 🤖 Trained TensorFlow/Keras model
* 🌐 Interactive Streamlit web application
* 📊 Prediction score
* 🔄 Reset functionality
* 🚀 Live deployment on Streamlit

---

## 🛠️ Tech Stack

| Technology           | Purpose                        |
| -------------------- | ------------------------------ |
| **Python**           | Programming language           |
| **TensorFlow**       | Deep learning framework        |
| **Keras**            | Neural network API             |
| **RNN**              | Sequential text classification |
| **IMDB Dataset**     | Sentiment analysis dataset     |
| **NumPy**            | Numerical operations           |
| **Streamlit**        | Web application                |
| **Jupyter Notebook** | Model development              |

---

## 📂 Project Structure

```text
RNN-review-analysis/
│
├── app.py
│   └── Streamlit web application
│
├── embedding.ipynb
│   └── Word embedding experiments
│
├── prediction.ipynb
│   └── Model prediction and testing
│
├── simplernn.ipynb
│   └── RNN model development and training
│
├── simple_rnn_imdb.h5
│   └── Trained RNN model
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

---

# 🧠 How It Works

## 1. IMDB Dataset

The project uses the **IMDB Movie Review Dataset** available through TensorFlow/Keras.

The reviews are labeled according to their sentiment:

```text
0 → Negative
1 → Positive
```

---

## 2. Text Preprocessing

Raw text cannot be directly passed to a neural network.

The review is first converted into a sequence of numerical word IDs.

For example:

```text
"This movie was amazing"
```

becomes something similar to:

```text
[15, 23, 8, 125]
```

The sequence is then padded to a fixed length:

```python
sequence.pad_sequences(
    [encoded_review],
    maxlen=500
)
```

This ensures that every input has the same shape.

---

## 3. Word Embedding

The numerical word representation is passed through an embedding layer.

The embedding layer converts word IDs into dense numerical vectors.

```text
Word ID
   ↓
Embedding Layer
   ↓
Dense Vector Representation
```

This allows the model to learn relationships between words.

---

## 4. Recurrent Neural Network

The embedded sequence is passed through an RNN.

RNNs are useful for sequential data because they process information while considering the order of the inputs.

```text
Word 1 → Word 2 → Word 3 → Word 4
   ↓        ↓        ↓        ↓
              RNN
               ↓
        Learned Representation
               ↓
        Classification Layer
```

---

## 5. Prediction

The final layer uses a sigmoid activation function and produces a score between `0` and `1`.

```text
0.0 ───────────────────────────── 1.0
 │                                  │
Negative                         Positive
```

The Streamlit application interprets the score using:

```text
Score <= 0.35         → Negative

0.35 < Score < 0.65  → Neutral

Score >= 0.65        → Positive
```

> **Note:** The RNN itself is trained as a binary sentiment classifier. The `Neutral` category is an application-level interpretation for predictions close to the decision boundary; it is not a separately trained third class.

---

# 🌐 Streamlit Application

The trained model is loaded by `app.py` and used to classify new movie reviews.

The application provides:

### 📝 Review Input

Users can enter a movie review into the text area.

### 🤖 Sentiment Classification

The review is processed and passed to the trained RNN model.

### 📊 Prediction Score

The model's output score is displayed to the user.

### 🔄 Reset

Users can clear the input and test another review.

---

# 🧪 Example

### Positive Review

```text
This movie was absolutely fantastic. The acting was excellent,
the story was engaging, and I loved every minute of it.
```

Possible result:

```text
Sentiment: Positive
```

---

### Negative Review

```text
This movie was extremely disappointing. The story was boring,
the acting was weak, and the ending made no sense.
```

Possible result:

```text
Sentiment: Negative
```

---

### Mixed Review

```text
The acting was excellent and the movie looked beautiful,
but unfortunately the story was incredibly boring and predictable.
I enjoyed some parts, but overall I was disappointed.
```

A model score around `0.60` would be interpreted by the application as:

```text
Sentiment: Neutral
Prediction Score: 0.6000
```

This demonstrates how a review containing both positive and negative signals can result in an uncertain prediction.

---

# ⚙️ Installation

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

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

in your browser.

---

# 📓 Notebooks

## `simplernn.ipynb`

Contains the main RNN development workflow, including:

* Dataset loading
* Data preparation
* Model creation
* Model training
* Model evaluation
* Model saving

## `embedding.ipynb`

Contains experiments and learning related to:

* Word embeddings
* Text representation
* Embedding layers

## `prediction.ipynb`

Contains:

* Loading the trained model
* Preparing new reviews
* Generating predictions
* Testing model behavior

---

# 📦 Trained Model

The trained model is stored in:

```text
simple_rnn_imdb.h5
```

The Streamlit application loads the model using TensorFlow/Keras:

```python
model = load_model("simple_rnn_imdb.h5")
```

---

# 📈 Future Improvements

Some possible improvements for this project include:

* [ ] Replace Simple RNN with LSTM
* [ ] Experiment with GRU
* [ ] Improve text preprocessing
* [ ] Add dropout and regularization
* [ ] Perform hyperparameter tuning
* [ ] Add confusion matrix
* [ ] Add classification report
* [ ] Add training/validation graphs
* [ ] Integrate TensorBoard
* [ ] Add prediction confidence visualization
* [ ] Support batch review prediction
* [ ] Train a dedicated three-class sentiment model
* [ ] Improve deployment and scalability

---

# 🎯 Learning Outcomes

This project provided practical experience with:

* Natural Language Processing
* Text preprocessing
* Tokenization
* Word indexing
* Sequence padding
* Word embeddings
* Recurrent Neural Networks
* Binary classification
* Sigmoid activation
* TensorFlow/Keras
* Model training and evaluation
* Streamlit application development
* Machine learning model deployment

---

# 🚀 Deployment

The application is deployed using **Streamlit**.

### Live Application

👉 **[Open RNN Review Analysis](https://rnn-review-analysis-oaaljqtwwe3dymse8tkez8.streamlit.app/)**

---

# 👨‍💻 Author

## Bharat Kumar

Data Science & Machine Learning Enthusiast


---

# ⭐ Show Your Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

Your support is appreciated!

---

## 📄 License

This project is developed for **educational and learning purposes**.
