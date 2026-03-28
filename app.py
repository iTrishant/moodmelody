import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import os
import logging
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

CLIENT_ID     = 'dc88dca2f2e549bbbdef653f12ddb042'
CLIENT_SECRET = '00e8ea8136994ceaada7b3833c88edcf'
REDIRECT_URI  = 'https://moodmelody-6zfdngomcsns4f6rhmwx6b.streamlit.app'

SCOPE = "user-modify-playback-state user-read-playback-state"

emotion_to_song_uri = {
    'anger'   : 'spotify:track:7iN1s7xHE4ifF5povM6A48',
    'fear'    : 'spotify:track:3KkXRkHbMCARz0aVfEt68P',
    'joy'     : 'spotify:track:7qiZfU4dY1lWllzX7mPBI3',
    'love'    : 'spotify:track:3d9DChrdc6BOeFsbrZ3Is0',
    'sadness' : 'spotify:track:008McaJl3WM1UqxxVie9BP',
    'surprise': 'spotify:track:10nyNJ6zNy2YVYLrcwLccB',
}

@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model('./saved_models/Emotion Recognition from text.h5')
    with open('./saved_models/tokenizer.pkl',     'rb') as f: tok = pickle.load(f)
    with open('./saved_models/label_encoder.pkl', 'rb') as f: le  = pickle.load(f)
    with open('./saved_models/maxlen.pkl',        'rb') as f: mx  = pickle.load(f)
    return m, tok, le, mx

def get_sp_oauth():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotifycache",
        open_browser=False,
    )

def normalized_sentence(sentence):
    sentence = sentence.lower()
    sentence = " ".join(w for w in sentence.split() if w not in stop_words)
    sentence = ''.join(c for c in sentence if not c.isdigit())
    sentence = re.sub(f'[{re.escape(string.punctuation)}]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    sentence = re.sub(r'https?://\S+|www\.\S+', '', sentence)
    sentence = " ".join(lemmatizer.lemmatize(w) for w in sentence.split())
    return sentence

def predict_emotion(text, model, tokenizer, le, maxlen):
    text = normalized_sentence(text)
    seq  = tokenizer.texts_to_sequences([text])
    seq  = pad_sequences(seq, maxlen=maxlen, truncating='pre')
    pred = model.predict(seq)
    return le.inverse_transform([np.argmax(pred, axis=1)[0]])[0]

def play_song(sp, emotion):
    uri     = emotion_to_song_uri.get(emotion)
    devices = sp.devices().get('devices', [])
    if not devices:
        st.warning("No active Spotify device found. Open Spotify on any device first.")
        return
    sp.start_playback(device_id=devices[0]['id'], uris=[uri])
    st.success(f"Playing song for {emotion}")

def main():
    st.title("MoodMelody: Emotion-based Music Recommender")

    try:
        model, tokenizer, le, maxlen = load_assets()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return

    sp_oauth = get_sp_oauth()

    params = st.query_params
    if "code" in params and "token_info" not in st.session_state:
        try:
            token_info = sp_oauth.get_access_token(
                params["code"], as_dict=True, check_cache=False
            )
            st.session_state["token_info"] = token_info
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Auth failed (code may have expired). Please re-authenticate.\n\n{e}")
            return

    if "token_info" in st.session_state:
        token_info = st.session_state["token_info"]
        if sp_oauth.is_token_expired(token_info):
            token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
            st.session_state["token_info"] = token_info
        sp = spotipy.Spotify(auth=token_info["access_token"])
    else:
        sp = None

    text = st.text_input("Enter how you are feeling:")

    if st.button("Detect Emotion and Play Song"):
        if not text.strip():
            st.warning("Please enter some text.")
            return

        emotion = predict_emotion(text, model, tokenizer, le, maxlen)
        st.write(f"Detected emotion: {emotion}")

        if sp is None:
            auth_url = sp_oauth.get_authorize_url()
            st.markdown(
                f'<a href="{auth_url}" target="_self">Click here to authenticate with Spotify</a>',
                unsafe_allow_html=True,
            )
            st.info("You will be redirected back automatically after authorising.")
        else:
            play_song(sp, emotion)

if __name__ == "__main__":
    main()
