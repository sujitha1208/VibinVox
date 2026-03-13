import streamlit as st
import tempfile
from transformers import pipeline
import librosa
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="VIBINVOX – EMOTION ENGINE", layout="centered")

# ---------- SUBTLE PREMIUM UI ----------
st.markdown("""
<style>

html, body, [class*="css"]{
font-family:"Times New Roman",serif;
}

.stApp{
background: linear-gradient(120deg,#f6f7fb,#eef1f6,#f6f7fb);
background-size:200% 200%;
animation: subtleMove 18s ease infinite;
color:#2b2b2b;
}

@keyframes subtleMove{
0%{background-position:0% 50%}
50%{background-position:100% 50%}
100%{background-position:0% 50%}
}

h1{
text-align:center;
font-size:40px;
letter-spacing:2px;
color:#1a1a1a;
}

.stButton>button{
background: linear-gradient(90deg,#ff9966,#ff5e62);
color:white;
border:none;
border-radius:8px;
height:45px;
width:220px;
font-size:17px;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1>VIBINVOX – EMOTION ENGINE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;'>Precision in Every Vibration</p>", unsafe_allow_html=True)

st.markdown("---")

st.info("🎙 Record or upload voice and VibinVox will analyze the emotion.")

# Emotion model
emotion_model = pipeline(
"audio-classification",
model="superb/wav2vec2-base-superb-er"
)

# Audio inputs
record = st.audio_input("🎙 Record your voice")
upload = st.file_uploader(
"Upload voice",
type=["wav","mp3","m4a","ogg","webm","flac"]
)

audio = record if record else upload

if audio:

    st.audio(audio)

    with st.spinner("Analyzing vibrations..."):
        time.sleep(2)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(audio.read())

    # -------- Waveform --------
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

        main = labels[0]
        emotion = emotion_map.get(main,"Neutral 😐")

        st.success(f"🎯 Detected Emotion: {emotion}")

        # -------- Confidence Bars --------
        fig2, ax2 = plt.subplots(figsize=(6,2))
        ax2.barh(labels, scores)
        ax2.set_xlim(0,1)
        ax2.set_title("Emotion Confidence")
        ax2.set_xlabel("Probability")

        st.pyplot(fig2)

        # -------- Emotion Animations --------

        if main == "hap":

            st.balloons()
            st.markdown("## 😄 Happiness Detected")
            st.markdown("💛 Your voice sounds joyful and energetic!")

        elif main == "sad":

            st.markdown("## 😢 Sadness Detected")
            st.markdown("🌧️ Your voice carries a sad tone.")
            st.markdown("💙 Sending calm and comfort.")

        elif main == "ang":

            st.markdown("## 😡 Anger Detected")
            st.markdown("🔥 Strong emotional tone detected.")
            st.markdown("🧘 Try taking a deep breath and relaxing.")

        elif main == "neu":

            st.markdown("## 😌 Neutral Emotion")
            st.markdown("🌿 Your voice sounds calm and balanced.")

        else:

            st.markdown("✨ Emotion detected successfully.")
