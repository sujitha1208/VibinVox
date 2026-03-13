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
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&display=swap');

html, body, [class*="css"]{
font-family:"Times New Roman",serif;
}

.stApp{
background: linear-gradient(135deg,#e0e0e0,#f2f2f2,#d8d8d8);
background-size:300% 300%;
animation: moveBG 20s ease infinite;
}

@keyframes moveBG{
0%{background-position:0% 50%}
50%{background-position:100% 50%}
100%{background-position:0% 50%}
}

.title{
font-family:'Playfair Display',serif;
font-size:50px;
text-align:center;
letter-spacing:3px;
color:#111;
}

.tag{
text-align:center;
font-size:18px;
color:#444;
letter-spacing:2px;
margin-top:-10px;
}

.card{
background:rgba(255,255,255,0.65);
padding:25px;
border-radius:14px;
backdrop-filter:blur(10px);
box-shadow:0 10px 25px rgba(0,0,0,0.08);
}

.stButton>button{
background:linear-gradient(90deg,#444,#222);
color:white;
border:none;
border-radius:8px;
height:45px;
width:220px;
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

    with st.spinner("Analyzing vibrations..."):
        time.sleep(2)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(audio.read())

    # -------- Animated Waveform --------
    y, sr = librosa.load(temp.name)

    fig, ax = plt.subplots(figsize=(6,1.5))
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

        # -------- Emotion Meter Gauge --------
        fig2, ax2 = plt.subplots(figsize=(4,4))

        value=scores[0]
        theta=np.linspace(0,np.pi,100)

        ax2.plot(np.cos(theta),np.sin(theta))
        ax2.scatter(np.cos(value*np.pi),np.sin(value*np.pi),s=200)

        ax2.set_title("Emotion Intensity Meter")
        ax2.axis("off")

        st.pyplot(fig2)

        # -------- Emotion Advice --------
        if main=="hap":

            st.balloons()
            st.markdown("### 😄 Happiness Detected")
            st.markdown("💛 Your voice radiates positivity and excitement.")
            st.markdown("✨ Keep spreading that positive energy!")

        elif main=="sad":

            st.markdown("### 😢 Sadness Detected")
            st.markdown("🌧 Your tone reflects sadness.")
            st.markdown("💙 Suggestion: take a break, talk to someone, or listen to music.")

        elif main=="ang":

            st.markdown("### 😡 Anger Detected")
            st.markdown("🔥 Strong emotional energy detected.")
            st.markdown("🧘 Suggestion: pause, breathe slowly, and relax your mind.")

        elif main=="neu":

            st.markdown("### 😌 Neutral Emotion")
            st.markdown("🌿 Your voice sounds calm and balanced.")
            st.markdown("✨ Maintain this peaceful state.")

st.markdown('</div>', unsafe_allow_html=True)
