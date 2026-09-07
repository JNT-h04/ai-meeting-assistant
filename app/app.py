import os
import tempfile

import streamlit as st
from meeting_model import process_audio

st.set_page_config(page_title="AI Meeting Assistant", layout="wide")

st.title("🎤 AI Meeting Assistant")
st.write("Upload audio → Get transcription, diarization & summary")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if uploaded_file is not None:

    st.info("Processing... please wait ⏳")

    # save temp file, keeping the original extension so downstream audio
    # libraries (which sniff by extension, not just content) decode it correctly
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        transcript, summary = process_audio(audio_path)

        st.subheader("📄 Transcript")
        st.text_area("", transcript, height=400)

        st.subheader("📌 Summary")
        st.write(summary)
    except Exception as e:
        st.error(f"Processing failed: {e}")
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)