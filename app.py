import streamlit as st
import os
from dotenv import load_dotenv
import groq
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import magic
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from docx import Document
from fpdf import FPDF
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import json
import noisereduce as nr
import numpy as np
from scipy.io import wavfile
from moviepy.editor import VideoFileClip
import google.oauth2.credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import base64
import requests
from urllib.parse import urlparse
import mimetypes
from collections import Counter
import re
from textblob import TextBlob
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import subprocess
from streamlit.components.v1 import html

# Configure page with enhanced UI
st.set_page_config(
    page_title="Meeting Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Meeting Assistant Pro\nAI-powered meeting analysis platform"
    }
)

# Add custom CSS for improved styling
def inject_custom_css():
    st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .st-bb {
            background-color: #f0f2f6;
        }
        .st-at {
            background-color: #ffffff;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(195deg, #42424a 0%, #191919 100%);
            color: white;
        }
        .sidebar .sidebar-content .stRadio label {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Download required NLTK data with progress
def download_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('sentiment/vader_lexicon', 'vader_lexicon'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    for i, (path, name) in enumerate(resources):
        try:
            nltk.data.find(path)
        except LookupError:
            status_text.text(f"Downloading {name}...")
            nltk.download(name, quiet=True)
            progress_bar.progress((i + 1) / len(resources))
    
    progress_bar.empty()
    status_text.empty()

download_nltk_resources()

# Enhanced spaCy loader with version check
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
        if spacy.__version__ < '3.0':
            raise Exception("spaCy version outdated. Please update to v3.x")
        return nlp
    except Exception as e:
        st.error(f"""
        Failed to load spaCy model: {str(e)}
        Please install with:
        ```bash
        python -m spacy download en_core_web_sm
        ```
        """)
        st.stop()

nlp = load_spacy_model()

# Improved Groq client initialization
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("""
        Groq API key not found. Please:
        1. Create .env file with GROQ_API_KEY=your_key
        2. Or set in Streamlit secrets
        """)
        st.stop()
    
    try:
        client = groq.Client(api_key=api_key)
        # Test connection with lightweight request
        client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        return client
    except Exception as e:
        st.error(f"Groq connection failed: {str(e)}")
        st.stop()

groq_client = get_groq_client()

# Enhanced Google Calendar integration
def get_google_calendar_service():
    if not os.path.exists('credentials.json'):
        st.error("""
        Google credentials missing. Please:
        1. Create Google Cloud Project
        2. Enable Calendar API
        3. Download credentials.json
        """)
        st.stop()
    
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', 
                ['https://www.googleapis.com/auth/calendar']
            )
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('calendar', 'v3', credentials=creds)

# Enhanced audio processing with progress
def process_audio(file_path):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Converting to WAV...")
    wav_path = convert_audio_to_wav(file_path)
    progress_bar.progress(25)
    
    status_text.text("Reducing noise...")
    cleaned_path = reduce_noise(wav_path)
    progress_bar.progress(50)
    
    status_text.text("Enhancing audio...")
    enhanced_path = enhance_audio(cleaned_path)
    progress_bar.progress(75)
    
    status_text.text("Finalizing...")
    progress_bar.progress(100)
    
    status_text.empty()
    progress_bar.empty()
    
    return enhanced_path

# New feature: Audio enhancement
def enhance_audio(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.low_pass_filter(3000).high_pass_filter(200)
    audio = audio.normalize()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file_path = temp_file.name
        audio.export(temp_file_path, format="wav")
    
    return temp_file_path

# Enhanced transcription with chunk processing
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    full_text = []
    
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        
        # Split audio into 1-minute chunks
        chunk_length = 60 * 1000  # milliseconds
        chunks = make_chunks(audio, chunk_length)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            status_text.text(f"Transcribing chunk {i+1}/{len(chunks)}...")
            progress_bar.progress((i + 1) / len(chunks))
            
            try:
                text = recognizer.recognize_google(chunk)
                full_text.append(text)
            except sr.UnknownValueError:
                st.warning(f"Chunk {i+1}: Audio could not be understood")
            except sr.RequestError as e:
                st.error(f"Chunk {i+1}: API unavailable; {str(e)}")
                break
        
        progress_bar.empty()
        status_text.empty()
    
    return " ".join(full_text)

# New feature: Interactive transcript editor
def show_transcript_editor(transcript):
    st.subheader("Transcript Editor")
    edited_transcript = st.text_area("Edit transcript:", value=transcript, height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Changes"):
            return edited_transcript
    with col2:
        if st.button("Discard Changes"):
            return transcript
    
    return edited_transcript

# Enhanced analytics dashboard without wordcloud
def create_enhanced_dashboard(analysis_data):
    tabs = st.tabs([
        "📊 Overview", "😊 Sentiment", "🗣️ Speakers", 
        "✅ Actions", "📈 Trends"
    ])
    
    with tabs[0]:
        st.subheader("Meeting Overview")
        cols = st.columns(4)
        cols[0].metric("Duration", f"{analysis_data['duration']}m")
        cols[1].metric("Speakers", analysis_data['speaker_count'])
        cols[2].metric("Action Items", len(analysis_data['action_items']))
        cols[3].metric("Sentiment", analysis_data['overall_sentiment'])
        
        st.plotly_chart(px.timeline(
            analysis_data['timeline'],
            x_start="start",
            x_end="end",
            y="speaker",
            color="sentiment",
            title="Meeting Timeline"
        ))
    
    with tabs[1]:
        st.subheader("Sentiment Analysis")
        fig = px.pie(
            analysis_data['sentiment_distribution'],
            names='sentiment',
            values='value',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig)
        
        st.plotly_chart(px.area(
            analysis_data['sentiment_over_time'],
            x='time',
            y='value',
            color='sentiment',
            title="Sentiment Over Time"
        ))

# New feature: Action items board
def show_action_board(action_items):
    st.subheader("Action Items Board")
    
    for i, item in enumerate(action_items.split('\n')):
        with st.expander(f"Action #{i+1}: {item.split(' - ')[0]}"):
            cols = st.columns(3)
            assignee = cols[0].text_input("Assignee", value=item.split(' - ')[1], key=f"assignee_{i}")
            deadline = cols[1].date_input("Deadline", key=f"deadline_{i}")
            status = cols[2].selectbox("Status", ["Not Started", "In Progress", "Completed"], key=f"status_{i}")
            
            if st.button(f"Update Action #{i+1}"):
                # Update logic here
                st.success("Action item updated!")

# Enhanced main function with onboarding
def main():
    st.title("🎙️ Meeting Assistant Pro")
    
    # Onboarding tour
    if 'initialized' not in st.session_state:
        with st.chat_message("assistant"):
            st.write("""
            Welcome to Meeting Assistant Pro! Here's how to start:
            1. Upload a meeting recording
            2. Wait for processing
            3. Explore insights & export results
            """)
        st.session_state.initialized = True
    
    # File upload section
    with st.expander("📤 Upload Meeting Recording", expanded=True):
        file = st.file_uploader(
            "Select audio/video file", 
            type=['mp3', 'wav', 'mp4', 'm4a', 'ogg', 'avi', 'mov'],
            help="Max size: 100MB"
        )
        
        url = st.text_input("Or enter recording URL:")
    
    # Processing pipeline
    if file or url:
        with st.status("Processing...", expanded=True) as status:
            st.write("Starting processing pipeline")
            
            try:
                # File handling
                if file:
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(file.read())
                    file_path = temp_file.name
                else:
                    file_path = download_meeting_recording(url)
                
                # Audio processing
                st.write("Enhancing audio...")
                audio_path = process_audio(file_path)
                
                # Transcription
                st.write("Transcribing content...")
                transcript = transcribe_audio(audio_path)
                
                # Analysis
                st.write("Generating insights...")
                summary = summarize_text(transcript)
                action_items = extract_action_items(transcript)
                sentiments = analyze_sentiment(transcript)
                
                status.update(label="Processing complete!", state="complete")
            
            except Exception as e:
                status.update(label="Processing failed", state="error")
                st.error(f"Error: {str(e)}")
                st.stop()
        
        # Show results
        with st.container():
            edited_transcript = show_transcript_editor(transcript)
            
            tab_summary, tab_actions, tab_analytics = st.tabs([
                "Summary", "Action Items", "Advanced Analytics"
            ])
            
            with tab_summary:
                st.subheader("AI Summary")
                st.write(summary)
                
                export_format = st.selectbox("Export format", ["PDF", "DOCX", "TXT"])
                if st.button("Export Summary"):
                    export_summary(summary, export_format)
            
            with tab_actions:
                show_action_board(action_items)
            
            with tab_analytics:
                analysis_data = generate_analysis_data(transcript, sentiments)
                create_enhanced_dashboard(analysis_data)

# Run the app
if __name__ == "__main__":
    main()
