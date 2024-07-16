import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st
import torch

# Ensure the necessary NLTK data is downloaded
nltk.data.path.append('./nltk_data')

# Load NLTK resources
try:
    stop_words = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('stopwords', download_dir='./nltk_data/corpora')
    nltk.download('wordnet', download_dir='./nltk_data/corpora')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

model_path = './saved_models/Emotion Recognition from text.h5'
tokenizer_path = './saved_models/tokenizer.pkl'
label_encoder_path = './saved_models/label_encoder.pkl'
maxlen_path = './saved_models/maxlen.pkl'

@st.cache
try:
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    with open(label_encoder_path, 'rb') as file:
        le = pickle.load(file)
    with open(maxlen_path, 'rb') as file:
        maxlen = pickle.load(file)
    st.success('Model and tokenizer loaded successfully!')
except OSError as e:
    st.error(f'Error loading model: {e}')

# Check if files exist
st.write(f"Model file exists: {os.path.exists(model_path)}")
st.write(f"Tokenizer file exists: {os.path.exists(tokenizer_path)}")
st.write(f"Label Encoder file exists: {os.path.exists(label_encoder_path)}")
st.write(f"Maxlen file exists: {os.path.exists(maxlen_path)}")

# Check file permissions
st.write(f"Model file permissions: {os.stat(model_path)}")

# Check file sizes
st.write(f"Model file size: {os.path.getsize(model_path)} bytes")
st.write(f"Tokenizer file size: {os.path.getsize(tokenizer_path)} bytes")
st.write(f"Label Encoder file size: {os.path.getsize(label_encoder_path)} bytes")
st.write(f"Maxlen file size: {os.path.getsize(maxlen_path)} bytes")

# Preprocessing steps
def lemmatization(text):
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def remove_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def remove_punctuations(text):
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = remove_numbers(sentence)
    sentence = remove_punctuations(sentence)
    sentence = remove_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

# Test prediction
text = "I am feeling very happy today!"
normalized_text = normalized_sentence(text)
sequence = tokenizer.texts_to_sequences([normalized_text])
padded_sequence = pad_sequences(sequence, maxlen=maxlen, truncating='pre')

try:
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_emotion = le.inverse_transform(predicted_label)[0]
    st.write(f"Predicted emotion: {predicted_emotion}")
except Exception as e:
    st.write(f"Error during prediction: {e}")

