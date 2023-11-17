# -*- coding: utf-8 -*-
import streamlit as st
import requests
import langid

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

# Set a modern color scheme
st.set_page_config(page_title="Speech-to-Text Transcription Using Whisper LV3", page_icon="🎙️", layout="wide", initial_sidebar_state="collapsed")

# Function to perform sentiment analysis using langid
def analyze_language(text):
    lang, _ = langid.rank(text)
    return lang

# Set a more modern title and subtitle
st.title("Whisper Large V3: Real-Time Speech-to-Text & Text Analytics 🎙️")
st.subheader("Upload an audio file in FLAC format to transcribe and analyze the text.")

# Create a file upload widget
audio_file = st.file_uploader("Choose an audio file (FLAC format)", type=["flac"])

if audio_file is not None:
    # Display audio playback controls
    st.audio(audio_file)

    if st.button("Transcribe & Analyze"):
        # Make a request to the Whisper API
        response = requests.post(API_URL, headers=headers, data=audio_file.read())

        if response.status_code == 200:
            result = response.json()
            if "text" in result:
                # Display the transcribed text
                st.success("Transcription Result:")
                transcribed_text = result["text"]
                st.write(transcribed_text)

                # Perform language identification
                language = analyze_language(transcribed_text)

                # Display language identification result
                st.subheader("Language Identification:")
                st.write(f"Detected Language: {language}")

            else:
                st.error("Transcription failed. Check the audio file format.")
        else:
            st.error(f"Request failed with status code: {response.status_code}")

# Provide additional information about the Whisper model
st.markdown(
    "Note: Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. "
    "Trained on 680k hours of labeled data, Whisper models demonstrate a strong ability to generalize to many datasets and domains without the need for fine-tuning."
)
