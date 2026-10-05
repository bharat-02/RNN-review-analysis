# Step 1: Import Libraries and Load the Model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')


# Step 2: Helper Functions

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in encoded_review]
    )


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


# Step 3: Streamlit App

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')

st.write(
    'Enter a movie review to classify it as positive or negative.'
)


# Step 4: Reset Function

def reset_review():
    st.session_state.review = ""


# Step 5: Movie Review Input

user_input = st.text_area(
    'Movie Review',
    key='review'
)


# Step 6: Classify Button

if st.button('Classify'):

    if user_input.strip() == "":
        st.warning('Please enter a movie review.')

    else:

        # Preprocess user input
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)

      # Determine sentiment
        score = prediction[0][0]
        if score >= 0.65:
            sentiment = "Positive"
        elif score <= 0.35:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"     


   



        # Display result
        st.write(f'Sentiment: {sentiment}')

        st.write(
            f'Prediction Score: {prediction[0][0]:.4f}'
        )


# Step 7: Reset Button

st.button(
    'Reset',
    on_click=reset_review
)