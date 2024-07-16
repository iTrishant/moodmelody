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
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@st.cache
def load_model():
    return torch.load('./saved_models/Emotion Recognition from text.h5')
def load_tokenizer():
    return torch.load('./saved_models/tokenizer.pkl')
def load_le():
    return torch.load('./saved_models/label_encoder.pkl')
def load_maxlen():
    return torch.load('./saved_models/maxlen.pkl')
    
@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
          
model_path = './saved_models/Emotion Recognition from text.h5'
tokenizer_path = './saved_models/tokenizer.pkl'
label_encoder_path = './saved_models/label_encoder.pkl'
maxlen_path = './saved_models/maxlen.pkl'

# Check if files exist
print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Tokenizer file exists: {os.path.exists(tokenizer_path)}")
print(f"Label Encoder file exists: {os.path.exists(label_encoder_path)}")
print(f"Maxlen file exists: {os.path.exists(maxlen_path)}")

# Check file permissions
print(f"Model file permissions: {os.stat(model_path)}")

# Check file sizes
print(f"Model file size: {os.path.getsize(model_path)} bytes")
print(f"Tokenizer file size: {os.path.getsize(tokenizer_path)} bytes")
print(f"Label Encoder file size: {os.path.getsize(label_encoder_path)} bytes")
print(f"Maxlen file size: {os.path.getsize(maxlen_path)} bytes")

model = load_model()
tokenizer = load_tokenizer()
maxlen = load_maxlen()
le = load_le()
    
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
    print(f"Predicted emotion: {predicted_emotion}")
except Exception as e:
    print(f"Error during prediction: {e}")

