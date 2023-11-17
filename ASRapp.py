import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

st.title("Real-Time Speech-to-Text with Whisper Large V3")

# Create a file upload widget
audio_file = st.file_uploader("Upload an audio file (in FLAC format)")

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Transcribe"):
        # Make a request to the Whisper API
        response = requests.post(API_URL, headers=headers, data=audio_file.read())

        if response.status_code == 200:
            result = response.json()
            if "text" in result:
                st.success("Transcription Result:")
                st.write(result["text"])
            else:
                st.error("Transcription failed. Check the audio file format.")
        else:
            st.error(f"Request failed with status code: {response.status_code}")

st.write("Note: Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours of labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains without the need for fine-tuning.")
