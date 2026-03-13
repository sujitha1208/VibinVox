import streamlit as st
import tempfile
from transformers import pipeline

st.set_page_config(page_title="VibinVox Emotion Engine")

st.title("🎤 VibinVox Emotion Engine")
st.write("Precision in every vibration")

emotion_model = pipeline(
"audio-classification",
model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

audio = st.file_uploader("Upload Voice", type=["wav","mp3"])

if audio:
    st.audio(audio)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(audio.read())

    if st.button("Analyze Emotion"):
        result = emotion_model(temp.name)
        emotion = result[0]["label"]
        st.success("Detected Emotion: " + emotion)
