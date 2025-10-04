# Automatic Speech Recognition (ASR) with Transformers

This project demonstrates a simple Automatic Speech Recognition (ASR) pipeline using the Hugging Face `transformers` library. It loads a pre-trained model to transcribe audio from a dataset sample.

## Overview

The script performs the following steps:
1.  Loads a pre-trained Wav2Vec2 model for speech recognition.
2.  Loads a sample audio file from a dataset.
3.  Processes the audio and prints the transcribed text.

## Requirements

Before running the code, ensure you have the following libraries installed:

```bash
pip install transformers datasets torchaudio
