# moodmelody

**Deployed App**: [MoodMelody on Streamlit](https://moodmelody-6zfdngomsns4f6rhmwx6b.streamlit.app/)

---

## Why I built this

I listen to music while studying, coding, and decompressing. The *“what should I play now?”* moment—those little music slumps—kept breaking my flow.  
I wanted something that understood how I felt and then just played music to match it.  

That became **moodmelody** — a small project where I stitched together **NLP**, a lightweight **classifier**, and the **Spotify API** to recommend tracks based on the emotion expressed in text.

---

## Technical Overview
The idea behind **MoodMelody** is to combine **NLP + Recommender Systems** to enhance user experience in music discovery.  
Users provide a **text prompt** (e.g., *"I feel anxious but hopeful"*), and the system:  
1. Preprocesses the text (cleaning, tokenization, embeddings).  
2. Uses a **BiLSTM classifier** to detect the underlying emotion.  
3. Maps the detected emotion to a relevant **playlist/song recommendation**.  

---

## ✨ Features
- Emotion detection from free-form text prompts.  
- Pre-trained **GloVe word embeddings** for better semantic understanding.  
- **BiLSTM classifier** for accurate emotion prediction.  
- Recommendation engine mapping emotions → curated playlists.  
- **Streamlit web app** for interactive usage.  

---

## 🔄 Project Workflow
1. **Dataset Preparation**  
   - Emotion-labelled text dataset used for training.  
   - Preprocessing: lowercasing, tokenization, stopword removal.  

2. **Embedding**  
   - GloVe vectors to represent words in dense form.  

3. **Model Training**  
   - BiLSTM model trained for multi-class emotion classification.  

4. **Prediction**  
   - User input text → same preprocessing → embedding → model → emotion label.  

5. **Recommendation**  
   - Emotion label mapped to predefined playlists (Spotify/YouTube links).  

---

## 🛠 Tech Stack
- **Programming Language**: Python  
- **Libraries**: TensorFlow / Keras, NLTK, NumPy, Pandas, Scikit-learn  
- **Embeddings**: GloVe  
- **Model**: BiLSTM  
- **Deployment**: Streamlit  

---

## A general overview
## What it does

- You type a thought, sentence, or paragraph.  
- The app predicts the **mood/emotion** behind your text.  
- It maps that mood to **Spotify recommendations** and shows playable tracks.  
- If you authenticate with Spotify, playback happens on **your account and devices**.  

---

## How it works 

### **1. Text → Mood**
- Preprocessing: tokenization, lowercasing, and basic cleanup.  
- A trained classifier (stored in `saved_models/`) predicts a mood label.  
- The model is intentionally lightweight so the app feels **snappy**.

### **2. Mood → Music**
- Each mood maps to a set of **search queries** and/or **audio feature hints**.  
- Using the Spotify Web API (via `spotipy`), I fetch tracks/playlists that align with that mood.  
- Think of it as *“semantic nudge + API curation”*—not perfect science, but perfect for v1.

### **3. Spotify Auth (why your Spotify opens, not mine)**
- The app uses **OAuth** through `spotipy.SpotifyOAuth`.  
- You log in once → Spotify issues an **access token tied to your account**.  
- Spotipy **caches and refreshes tokens** automatically.  
- My client credentials only identify the app; **all playback is tied to your Spotify account**.

### **4. App Shell**
- Built with **Streamlit** for:
  - quick UI  
  - easy deployment  
  - frictionless user experience  

---

## Folder Structure
moodmelody/
├─ app.py                         # Streamlit app (main entry point)
├─ moodmelody_final_code.ipynb    # Experiments / final notebook
├─ saved_models/                  # Trained model(s) and artifacts
├─ nltk_data/                     # Local NLP resources (if needed)
├─ test.py                        # Small test harnesses
├─ requirements.txt               # Python dependencies
└─ README.md                     
