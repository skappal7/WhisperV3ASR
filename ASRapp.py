import streamlit as st
import requests
import spacy
import langid

# Set a modern color scheme
st.set_page_config(page_title="Speech-to-Text & Text Analytics", page_icon="🎙️", layout="wide", initial_sidebar_state="collapsed")

# Load spaCy models for English and Hindi
nlp_en = spacy.load("en_core_web_sm")
nlp_hi = spacy.load("hi_core_web_sm")

# Function to identify language and perform sentiment analysis
def analyze_sentiment_multilingual(text):
    segments = langid.rank(text)
    
    sentiment_results = []
    for segment in segments:
        lang_code, _ = segment
        if lang_code == 'en':
            doc = nlp_en(text)
        elif lang_code == 'hi':
            doc = nlp_hi(text)
        else:
            doc = None
        
        if doc:
            sentiment = doc.cats.get('positive', 0.0) - doc.cats.get('negative', 0.0)
            sentiment_results.append(sentiment)

    return sentiment_results

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

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

                # Perform sentiment analysis for each language segment
                sentiment_results = analyze_sentiment_multilingual(transcribed_text)

                # Display sentiment analysis results
                st.subheader("Sentiment Analysis Results:")
                for i, sentiment in enumerate(sentiment_results):
                    lang_code, _ = langid.rank(transcribed_text)[i]
                    lang_name = "English" if lang_code == "en" else "Hindi"
                    st.write(f"{lang_name} Segment {i + 1}: Sentiment Score: {sentiment:.2f}")

            else:
                st.error("Transcription failed. Check the audio file format.")
        else:
            st.error(f"Request failed with status code: {response.status_code}")

# Provide additional information about the Whisper model
st.markdown(
    "Note: Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. "
    "Trained on 680k hours of labeled data, Whisper models demonstrate a strong ability to generalize to many datasets and domains without the need for fine-tuning."
)
