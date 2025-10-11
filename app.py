import streamlit as st
from transformers import pipeline
import torchaudio
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf

# ---------------------------------------------------
# 1. Page setup
# ---------------------------------------------------
st.set_page_config(page_title="Speech-to-Text App", page_icon="🎤", layout="centered")
st.title("🎙️ Speech-to-Text App")
st.write("Record or upload an audio file to transcribe it into text using Wav2Vec2.")

# ---------------------------------------------------
# 2. Load ASR model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        task="automatic-speech-recognition",
        model="facebook/wav2vec2-large-960h"
    )

asr = load_model()

# Force torchaudio to use soundfile backend
torchaudio.set_audio_backend("soundfile")

# ---------------------------------------------------
# 3. Choose input type
# ---------------------------------------------------
st.subheader("Choose Audio Source")
option = st.radio("Select an option:", ["🎧 Upload Audio", "🎤 Record from Microphone"])

temp_path = None

# ---------------------------------------------------
# 4. Upload Audio Mode
# ---------------------------------------------------
if option == "🎧 Upload Audio":
    uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "flac", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

# ---------------------------------------------------
# 5. Record Audio Mode
# ---------------------------------------------------
elif option == "🎤 Record from Microphone":
    st.write("Press 'Start Recording' to begin and 'Stop Recording' to end.")
    sample_rate = 16000

    # Initialize session states
    if "recording" not in st.session_state:
        st.session_state.recording = False
        st.session_state.audio_data = None

    # Start Recording
    if not st.session_state.recording:
        if st.button("🎙️ Start Recording"):
            st.session_state.recording = True
            st.session_state.audio_data = sd.rec(
                int(60 * sample_rate),  # up to 60s buffer
                samplerate=sample_rate,
                channels=1,
                dtype="float32"
            )
            st.info("Recording... 🎤 Speak now!")
    else:
        # Stop Recording
        if st.button("⏹️ Stop Recording"):
            sd.stop()
            st.session_state.recording = False
            st.success("✅ Recording stopped!")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, st.session_state.audio_data, sample_rate)
                temp_path = tmp.name

            st.audio(temp_path, format="audio/wav")

# ---------------------------------------------------
# 6. Run ASR transcription
# ---------------------------------------------------
if temp_path:
    try:
        waveform, sr = torchaudio.load(temp_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        st.write("🔍 Transcribing, please wait...")
        result = asr(waveform.squeeze().numpy())
        st.success("✅ Transcription complete!")

        st.text_area("Transcribed Text", result["text"], height=200)
        st.download_button("💾 Download Transcription", result["text"], file_name="transcription.txt")

    except Exception as e:
        st.error(f"⚠️ Error processing audio: {e}")
