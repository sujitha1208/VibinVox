import streamlit as st
import tempfile
from transformers import pipeline
import librosa
import matplotlib.pyplot as plt
import numpy as np
import time

st.set_page_config(page_title="VIBINVOX – EMOTION ENGINE", layout="centered")

# ---------- PREMIUM UI ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Poppins:wght@300;500&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"]{
font-family:'Poppins',sans-serif;
}

.stApp{
background: linear-gradient(120deg,#fdfbfb,#e9eef3);
}

h1{
font-family:'Orbitron',sans-serif;
text-align:center;
letter-spacing:2px;
}

.card{
background: rgba(255,255,255,0.6);
padding:20px;
border-radius:12px;
backdrop-filter: blur(12px);
}

.stButton>button{
background: linear-gradient(90deg,#ff7a18,#ffb347);
color:white;
border:none;
border-radius:10px;
height:45px;
width:220px;
font-size:16px;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1>VIBINVOX – EMOTION ENGINE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Precision in Every Vibration</p>", unsafe_allow_html=True)

st.markdown("---")

st.info("🎙 Record or upload your voice to detect emotion.")

# Emotion model
emotion_model = pipeline(
"audio-classification",
model="superb/wav2vec2-base-superb-er"
)

# ---- Audio Inputs ----
audio_record = st.audio_input("🎙 Record your voice")
audio_upload = st.file_uploader(
"Upload voice",
type=["wav","mp3","m4a","ogg","webm","flac"]
)

audio_file = audio_record if audio_record else audio_upload

if audio_file:

    st.audio(audio_file)

    # animated mic indicator
    with st.spinner("🎙 Processing Voice..."):
        time.sleep(2)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(audio_file.read())

    # waveform
    y, sr = librosa.load(temp.name)

    fig, ax = plt.subplots(figsize=(6,1.2))
    ax.plot(y)
    ax.axis("off")
    ax.set_title("Voice Waveform", fontsize=9)

    st.pyplot(fig)

    if st.button("Analyze Emotion 🎧"):

        results = emotion_model(temp.name)

        labels = [r["label"] for r in results]
        scores = [r["score"] for r in results]

        emotion_map = {
            "hap": "Happy 😊",
            "sad": "Sad 😢",
            "ang": "Angry 😡",
            "neu": "Neutral 😐"
        }

        main_code = labels[0]
        emotion = emotion_map.get(main_code,"Neutral 😐")

        st.success(f"🎯 Detected Emotion: {emotion}")

        # circular emotion meter
        fig2, ax2 = plt.subplots(figsize=(3,3))
        ax2.pie(scores, labels=labels, autopct='%1.0f%%')
        ax2.set_title("Emotion Distribution")

        st.pyplot(fig2)

        # AI emotional response
        if main_code == "hap":
            st.balloons()
            st.markdown("💛 **Your voice sounds joyful! Keep spreading positivity.**")

        elif main_code == "sad":
            st.markdown("💙 **You seem a bit sad. Maybe listen to music or talk to a friend.**")

        elif main_code == "ang":
            st.markdown("❤️ **Some anger detected. Try taking a deep breath and relaxing.**")

        elif main_code == "neu":
            st.markdown("🤍 **Your voice sounds calm and balanced.**")

        else:
            st.markdown("✨ Emotion detected successfully by VibinVox AI.")
