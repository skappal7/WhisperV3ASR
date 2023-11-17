import streamlit as st
import requests

# Set the page title and icon
st.set_page_config(page_title="Whisper ASR", page_icon="🔊", layout="centered")

# API details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

# Set the app title and a brief description
st.title("Real-Time Speech-to-Text with Whisper Large V3")
st.markdown("This app uses the Whisper Large V3 model for real-time automatic speech recognition (ASR). Upload an audio file in FLAC format and click the button to transcribe.")

# Create a file upload widget
audio_file = st.file_uploader("Upload an audio file (in FLAC format)", type=["flac"])

if audio_file is not None:
    st.audio(audio_file)

    # Improve the button appearance
    if st.button("Transcribe", key="transcribe_button", help="Click to transcribe the audio"):
        # Make a request to the Whisper API
        response = requests.post(API_URL, headers=headers, data=audio_file.read())

        if response.status_code == 200:
            result = response.json()
            if "text" in result:
                transcribed_text = result["text"]
                st.success("Transcription Result:")
                st.write(transcribed_text)
            else:
                st.error("Transcription failed. Check the audio file format.")
        else:
            st.error(f"Request failed with status code: {response.status_code}")

# Add a footer with additional information about the Whisper model
st.markdown(
    """
    ---\n
    **Note:** Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. 
    Trained on 680k hours of labeled data, Whisper models demonstrate a strong ability to generalize to many datasets and domains without the need for fine-tuning.
    """
)
