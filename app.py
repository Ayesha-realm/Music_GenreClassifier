import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import os
import tempfile

# ====================== CONFIG ======================
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load model (best to use Git LFS for large .h5 files)
@st.cache_resource
def load_classifier():
    model_path = "your_model.h5"   # Put your .h5 in the repo root (use Git LFS)
    return load_model(model_path)

model = load_classifier()

# ===================================================

st.title("🎵 Music Genre Classifier")
st.markdown("Upload a **30-second audio clip** and get the predicted genre!")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3, .ogg)", 
                                type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    st.audio(uploaded_file, format=uploaded_file.type)
    
    with st.spinner("Analyzing audio..."):
        try:
            # Preprocess (adjust according to your model's training)
            y, sr = librosa.load(temp_path, sr=22050, duration=30)
            
            # Mel spectrogram (CHANGE if your model uses different features)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Resize to model's expected input (example: 128x128)
            mel_spec_db = tf.image.resize(mel_spec_db[..., np.newaxis], (128, 128))
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)  # batch dimension

            # Predict
            predictions = model.predict(mel_spec_db, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx] * 100)
            predicted_genre = GENRES[predicted_idx]

            # Results
            st.success(f"**Predicted Genre: {predicted_genre.upper()}**")
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Show probabilities bar chart
            probs = predictions[0]
            chart_data = {GENRES[i]: float(probs[i]*100) for i in range(len(GENRES))}
            st.bar_chart(chart_data)

        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            os.unlink(temp_path)  # Clean up temp file
