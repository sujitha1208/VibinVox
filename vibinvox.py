import streamlit as st
import tempfile
from transformers import pipeline

st.set_page_config(page_title="VibinVox Emotion Engine", layout="centered")

# 🌈 Colorful UI
st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#667eea,#764ba2);
color:white;
}

h1 {
text-align:center;
}

.stButton>button {
background-color:#ff6a00;
color:white;
border-radius:12px;
height:50px;
width:220px;
font-size:18px;
}

.stFileUploader {
background-color:white;
padding:10px;
border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎤 VibinVox Emotion Engine")
st.write("### Precision in Every Vibration")

st.info("Upload or record a voice clip and VibinVox will detect emotion.")

# Load emotion AI model
emotion_model = pipeline(
"audio-classification",
model="superb/wav2vec2-base-superb-er"
)

audio = st.file_uploader("🎧 Upload Voice", type=["wav","mp3"])

if audio:

    st.audio(audio)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(audio.read())

    if st.button("Analyze Emotion 🎙"):

        result = emotion_model(temp.name)
        emotion = result[0]["label"]

        st.success("🎯 Detected Emotion: " + emotion)

        # Emotion based messages
        if emotion == "happy":
            st.balloons()
            st.markdown("💛 **Your voice sounds joyful and energetic! Keep smiling!**")

        elif emotion == "sad":
            st.markdown("💙 **Your tone feels a bit sad. Remember you're stronger than you think.**")

        elif emotion == "angry":
            st.markdown("❤️ **Some anger detected. Maybe take a deep breath and relax.**")

        elif emotion == "neutral":
            st.markdown("🤍 **Your voice is calm and balanced.**")

        else:
            st.markdown("✨ **Emotion successfully detected by VibinVox AI.**")
