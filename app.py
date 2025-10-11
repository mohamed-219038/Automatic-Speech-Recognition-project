import streamlit as st
from transformers import pipeline
import torch
import torchaudio
import tempfile
import numpy as np
import soundfile as sf

# ---------------------------------------------------
# 1. Page setup
# ---------------------------------------------------
st.set_page_config(page_title="Speech-to-Text App", page_icon="🎤", layout="centered")
st.title("🎙️ Speech-to-Text App")
st.write("Upload an audio file to transcribe it into text using Wav2Vec2.")

# ---------------------------------------------------
# 2. Load ASR model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        task="automatic-speech-recognition",
        model="facebook/wav2vec2-base-960h"  # Using base model for faster loading
    )

try:
    asr = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Force torchaudio to use soundfile backend
torchaudio.set_audio_backend("soundfile")

# ---------------------------------------------------
# 3. File upload section
# ---------------------------------------------------
st.subheader("Upload Audio File")
uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "flac", "m4a", "ogg"])

temp_path = None

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")
    
    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

# ---------------------------------------------------
# 4. Run ASR transcription
# ---------------------------------------------------
if temp_path:
    try:
        with st.spinner("🔍 Transcribing, please wait..."):
            # Load and preprocess audio
            waveform, sr = torchaudio.load(temp_path)
            
            # Resample if necessary
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Perform transcription
            result = asr(waveform.squeeze().numpy())
        
        st.success("✅ Transcription complete!")
        st.text_area("Transcribed Text", result["text"], height=200)
        
        # Download button
        st.download_button(
            "💾 Download Transcription", 
            result["text"], 
            file_name="transcription.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"⚠️ Error processing audio: {e}")
        st.info("💡 Try uploading a different audio file format (WAV, MP3, FLAC)")

# ---------------------------------------------------
# 5. Additional information
# ---------------------------------------------------
st.markdown("---")
st.subheader("ℹ️ Instructions")
st.markdown("""
1. Upload an audio file (WAV, MP3, FLAC, M4A, or OGG format)
2. Wait for the transcription to complete
3. Copy the text or download it as a file

**Note:** For best results, use clear audio with minimal background noise.
""")
