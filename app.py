
import streamlit as st
import torch
import numpy as np
import sqlite3
from datetime import datetime
from PIL import Image
import cv2
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Deception Detection", layout="centered")
st.title("🕵️ Deception Detection App")

# --- Database Setup ---
conn = sqlite3.connect('results.db')
c = conn.cursor()

# Tables for image and text detections
c.execute('''
CREATE TABLE IF NOT EXISTS image_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    detected_forgery INTEGER,
    timestamp TEXT
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS news_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_text TEXT,
    is_fake INTEGER,
    timestamp TEXT
)
''')
conn.commit()

def save_image_result(filename, is_forgery):
    now = datetime.now().isoformat()
    c.execute('INSERT INTO image_detections (filename, detected_forgery, timestamp) VALUES (?, ?, ?)',
              (filename, int(is_forgery), now))
    conn.commit()

def save_news_result(news_text, is_fake):
    now = datetime.now().isoformat()
    c.execute('INSERT INTO news_detections (news_text, is_fake, timestamp) VALUES (?, ?, ?)',
              (news_text[:1000], int(is_fake), now))
    conn.commit()

# Dummy image forgery detection (replace with actual ManTra-Net)
def dummy_detect_image(image_array):
    return np.random.choice([0, 1])  # Random prediction

# Dummy fake news detection model (replace with real model)
@st.cache_resource
def load_fake_news_model():
    # Simulate a trained model using TF-IDF + LogisticRegression
    return joblib.load("fake_news_model.pkl"), joblib.load("tfidf_vectorizer.pkl")

# --- App Layout with Tabs ---
tab1, tab2 = st.tabs(["🖼️ Image Forgery Detection", "📰 Fake News Detection"])

with tab1:
    st.header("Image Forgery Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True, caption="Uploaded Image")
        image_np = np.array(image)
        result = dummy_detect_image(image_np)
        st.success("Forgery Detected ✅" if result else "No Forgery Detected ❌")
        save_image_result(uploaded_file.name, result)

        st.subheader("Detection History")
        rows = c.execute('SELECT * FROM image_detections ORDER BY timestamp DESC').fetchall()
        for row in rows:
            st.write(f"📁 {row[1]} — Forgery: {'Yes' if row[2] else 'No'} — 🕒 {row[3]}")

with tab2:
    st.header("Fake News Detection")
    model, vectorizer = load_fake_news_model()
    news_text = st.text_area("Paste news article text below:")
    file = st.file_uploader("Or upload a text file", type=["txt"])

    if file:
        news_text = file.read().decode("utf-8")
        st.text_area("Extracted Text", value=news_text, height=200)

    if news_text:
        X = vectorizer.transform([news_text])
        prediction = model.predict(X)[0]
        st.success("🔴 Fake News Detected!" if prediction == 1 else "🟢 This seems to be Real News")
        save_news_result(news_text, prediction)

        st.subheader("Detection History")
        rows = c.execute('SELECT * FROM news_detections ORDER BY timestamp DESC').fetchall()
        for row in rows:
            st.write(f"📰 {'Fake' if row[2] else 'Real'} — 🕒 {row[3]}")

