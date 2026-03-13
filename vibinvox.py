import streamlit as st
import tempfile
from transformers import pipeline
import librosa
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="VibinVox Emotion Engine", layout="centered")

# ---- Subtle UI ----
st.markdown("""
<style>
.stApp{
background-color:#f5f5f5;
}

h1{
text-align:center;
color:#333;
}

.stButton>button{
background-color:#ff7a18;
color:white;
border-radius:8px;
height:45px;
width:200px;
font-size:16px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎤 VibinVox Emotion Engine")
st.write("Precision in every vibration")

st.info("Record or upload voice to detect emotion.")

# Emotion model
emotion_model = pipeline(
"audio-classification",
model="superb/wav2vec2-base-superb-er"
)

# ---- Audio Input ----
audio_record = st.audio_input("🎙 Record your voice")
audio_upload = st.file_uploader(
"Upload voice",
type=["wav","mp3","m4a","ogg","webm","flac"]
)

audio_file = audio_record if audio_record else audio_upload

if audio_file:

    st.audio(audio_file)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(audio_file.read())

    # ---- Waveform visualization ----
    y, sr = librosa.load(temp.name)

    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Voice Waveform")
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

        main_emotion_code = labels[0]
        main_emotion = emotion_map.get(main_emotion_code,"Neutral 😐")

        st.success(f"🎯 Detected Emotion: {main_emotion}")

        # ---- Probability chart ----
        fig2, ax2 = plt.subplots()
        ax2.bar(labels, scores)
        ax2.set_title("Emotion Confidence Levels")
        ax2.set_ylabel("Confidence")
        st.pyplot(fig2)

        # ---- AI Mood Message ----
        if main_emotion_code == "hap":
            st.balloons()
            st.markdown("💛 You sound joyful! Keep spreading positivity.")

        elif main_emotion_code == "sad":
            st.markdown("💙 Your voice sounds a little sad. Hope things get better.")

        elif main_emotion_code == "ang":
            st.markdown("❤️ Some anger detected. Maybe take a deep breath.")

        elif main_emotion_code == "neu":
            st.markdown("🤍 Your voice seems calm and balanced.")

        else:
            st.markdown("✨ Emotion successfully detected by VibinVox AI.")
