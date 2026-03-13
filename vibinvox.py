import streamlit as st
import tempfile
from transformers import pipeline
import librosa
import matplotlib.pyplot as plt
import time
import numpy as np

st.set_page_config(page_title="VIBINVOX – EMOTION ENGINE", layout="centered")

# ---------- UI ----------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&display=swap');

html, body, [class*="css"]{
font-family:"Times New Roman",serif;
}

/* Grey elegant background */

.stApp{
background: linear-gradient(135deg,#e6e6e6,#f3f3f3,#e0e0e0);
background-size:300% 300%;
animation:bgMove 20s ease infinite;
}

@keyframes bgMove{
0%{background-position:0% 50%}
50%{background-position:100% 50%}
100%{background-position:0% 50%}
}

/* Title */

.title{
font-family:'Playfair Display',serif;
font-size:52px;
font-weight:900;
text-align:center;
letter-spacing:2px;
white-space:nowrap;
color:#111;
}

/* Tagline */

.tag{
text-align:center;
font-size:18px;
letter-spacing:2px;
color:#555;
margin-top:-8px;
}

/* Glass card */

.card{
background:rgba(255,255,255,0.7);
padding:25px;
border-radius:14px;
box-shadow:0 8px 25px rgba(0,0,0,0.08);
}

/* Button */

.stButton>button{
background:linear-gradient(90deg,#444,#222);
color:white;
border:none;
border-radius:8px;
height:45px;
width:220px;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">VIBINVOX – EMOTION ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="tag">Precision in Every Vibration</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

st.info("🎙 Record or upload voice and VibinVox will analyze the emotional vibration.")

# Emotion model
emotion_model = pipeline(
"audio-classification",
model="superb/wav2vec2-base-superb-er"
)

record = st.audio_input("🎙 Record voice")
upload = st.file_uploader("Upload voice",type=["wav","mp3","m4a","ogg","webm","flac"])

audio = record if record else upload

if audio:

    st.audio(audio)

    with st.spinner("Analyzing voice vibrations..."):
        time.sleep(2)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(audio.read())

    # -------- WAVEFORM --------
    y, sr = librosa.load(temp.name)

    fig, ax = plt.subplots(figsize=(6,1.4))
    ax.plot(y,color="black")
    ax.set_title("Voice Frequency Pattern",fontsize=10)
    ax.axis("off")

    st.pyplot(fig)

    if st.button("Analyze Emotion 🎧"):

        results = emotion_model(temp.name)

        labels=[r["label"] for r in results]
        scores=[r["score"] for r in results]

        emotion_map={
        "hap":"Happy 😊",
        "sad":"Sad 😢",
        "ang":"Angry 😡",
        "neu":"Neutral 😐"
        }

        main=labels[0]
        emotion=emotion_map.get(main,"Neutral 😐")

        st.success(f"🎯 Detected Emotion: {emotion}")

        # -------- Emotion Confidence Graph --------
        fig2, ax2 = plt.subplots(figsize=(6,2))

        ax2.barh(labels,scores)
        ax2.set_xlim(0,1)
        ax2.set_xlabel("Confidence Level")
        ax2.set_title("Emotion Confidence Analysis")

        st.pyplot(fig2)

        # -------- Emotion Responses --------

        if main=="hap":

            st.balloons()
            st.markdown("### 😄 Happiness Detected")
            st.markdown("💛 Your voice radiates joy and positivity.")
            st.markdown("✨ VibinVox Suggestion: Keep spreading that energy!")

        elif main=="sad":

            st.markdown("### 😢 Sadness Detected")
            st.markdown("🌧 Your tone reflects sadness.")
            st.markdown("💙 VibinVox Suggestion: Take a break or listen to calming music.")

        elif main=="ang":

            st.markdown("### 😡 Anger Detected")
            st.markdown("🔥 Strong emotional tone detected.")
            st.markdown("🧘 VibinVox Suggestion: Pause and take slow deep breaths.")

        elif main=="neu":

            st.markdown("### 😌 Neutral Emotion")
            st.markdown("🌿 Your voice sounds calm and balanced.")
            st.markdown("✨ VibinVox Suggestion: Maintain this peaceful tone.")

st.markdown('</div>', unsafe_allow_html=True)
