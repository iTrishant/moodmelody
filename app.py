import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.data.path.append("./nltk_data")
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords', download_dir='./nltk_data')
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', download_dir='./nltk_data')
  
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Replace with your Spotify app credentials
CLIENT_ID = 'dc88dca2f2e549bbbdef653f12ddb042'
CLIENT_SECRET = '00e8ea8136994ceaada7b3833c88edcf'
REDIRECT_URI = 'http://localhost:5000/callback'

# Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope="user-modify-playback-state user-read-playback-state",
                                               open_browser=False,
                                               cache_path="./.spotifycache"))

sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope="user-modify-playback-state user-read-playback-state",
                        cache_path="./.spotifycache")

# Define song URIs for each emotion
emotion_to_song_uri = {
    'anger': 'spotify:track:7iN1s7xHE4ifF5povM6A48',    # let it be
    'fear': 'spotify:track:3KkXRkHbMCARz0aVfEt68P',     # sunflower
    'joy': 'spotify:track:7qiZfU4dY1lWllzX7mPBI3',      # shape of you
    'love': 'spotify:track:3d9DChrdc6BOeFsbrZ3Is0',     # under the bridge
    'sadness': 'spotify:track:008McaJl3WM1UqxxVie9BP',  # the wisp sings
    'surprise': 'spotify:track:10nyNJ6zNy2YVYLrcwLccB'  # no surprises
}

# Load the saved model, tokenizer, label encoder, and maxlen
model_path = './saved_models/Emotion Recognition from text.h5'
try:
    model = tf.keras.models.load_model(model_path)
    st.success('Model loaded successfully!')
except OSError as e:
    st.error(f'Error loading model: {e}')
  
with open('./saved_models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open('./saved_models/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)
with open('./saved_models/maxlen.pkl', 'rb') as file:
    maxlen = pickle.load(file)

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

# Function to preprocess and predict emotion
def predict_emotion(text):
    # Preprocessing
    text = normalized_sentence(text)
    #st.write(f"Preprocessed text: {text}")
    sequence = tokenizer.texts_to_sequences([text])
    #st.write(f"Tokenized sequence: {sequence}")
    padded_sequence = pad_sequences(sequence, maxlen=maxlen, truncating='pre')
    #st.write(f"Padded sequence: {padded_sequence}")

    # Predict emotion
    prediction = model.predict(padded_sequence)
    #st.write(f"Prediction probabilities: {prediction}")
    predicted_label = np.argmax(prediction, axis=1)
    #st.write(f"Predicted label index: {predicted_label}")
    predicted_emotion = le.inverse_transform(predicted_label)[0]

    return predicted_emotion

# Function to play song based on emotion
def play_song(emotion):
    song_uri = emotion_to_song_uri.get(emotion)
    if song_uri:
        devices = sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']  # Select the first available device
            sp.start_playback(device_id=device_id, uris=[song_uri])
            st.write(f"Playing song for {emotion}: {song_uri}")
        else:
            st.write("No active devices found.")
    else:
        st.write("No song found for this emotion.")

# refreshing tokens in case of expiry
def refresh_spotify_token():
    token_info = sp_oauth.get_cached_token()
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
    return token_info['access_token']

def get_spotify_auth_url():
    return sp_oauth.get_authorize_url()
    
# Main function for Streamlit app
def main():
    st.title("MoodMelody: Emotion-based Music Recommender")
    # Checking user authentication
    if not sp_oauth.get_cached_token():
        st.error("Please authorize MoodMelody to access your Spotify account.")
        auth_url = get_spotify_auth_url()
        st.markdown(f"[Authorize MoodMelody]({auth_url})")
        return

    # Authorized user flow
    text = st.text_input("Enter how you are feeling:")
    
    if st.button("Detect Emotion and Play Song"):
        detected_emotion = predict_emotion(text)
        st.write(f"Detected emotion: {detected_emotion}")
        play_song(detected_emotion)

if __name__ == "__main__":
    main()
