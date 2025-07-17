# app.py

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load model and tokenizer
model = load_model("cnn_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit UI
st.title("📊 Indonesian Political Sentiment Classifier")
st.markdown("Enter a political comment to detect its emotional tone using deep learning.")

user_input = st.text_area("📝 Enter a comment", height=150)

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded)[0]

    predicted_labels = [label for label, score in zip(mlb.classes_, prediction) if score > 0.5]

    if predicted_labels:
        st.subheader("🎯 Predicted Emotions:")
        st.success(", ".join(predicted_labels))
    else:
        st.warning("No strong emotion detected.")
