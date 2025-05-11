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
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
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

# Configure page
st.set_page_config(
    page_title="Meeting Summarizer AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

# Configure Groq
try:
    # Try to get API key from Streamlit secrets
    groq_client = groq.Groq(api_key=st.secrets["groq"]["api_key"])
except:
    # Fallback to environment variable
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Google Calendar API setup
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_google_calendar_service():
    """Set up Google Calendar API service."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('calendar', 'v3', credentials=creds)

def extract_audio_from_video(video_file):
    """Extract audio from video file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_audio_path = temp_audio.name
    
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(temp_audio_path)
    return temp_audio_path

def reduce_noise(audio_file):
    """Reduce background noise from audio file."""
    # Read audio file
    rate, data = wavfile.read(audio_file)
    
    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    
    # Save processed audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file_path = temp_file.name
    wavfile.write(temp_file_path, rate, reduced_noise)
    
    return temp_file_path

def analyze_speaking_time(transcription, speaker_segments):
    """Analyze speaking time per speaker."""
    speaking_time = {}
    for speaker, segments in speaker_segments.items():
        total_time = sum(end - start for start, end in segments)
        speaking_time[speaker] = total_time
    
    return speaking_time

def create_analytics_dashboard(transcription, sentiments, speaking_time, action_items):
    """Create comprehensive analytics dashboard."""
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sentiment Analysis", "Speaking Time", "Action Items"])
    
    with tab1:
        st.subheader("Meeting Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Meeting Duration", f"{sum(speaking_time.values()):.2f} seconds")
        with col2:
            st.metric("Number of Speakers", len(speaking_time))
        with col3:
            st.metric("Action Items", len(action_items.split('\n')))
    
    with tab2:
        st.subheader("Sentiment Analysis")
        df = pd.DataFrame(sentiments)
        fig = px.line(df, y=['positive', 'negative', 'neutral'], 
                     title='Sentiment Analysis Over Time')
        st.plotly_chart(fig)
        
        # Sentiment distribution pie chart
        avg_sentiment = df[['positive', 'negative', 'neutral']].mean()
        fig = px.pie(values=avg_sentiment.values, 
                    names=avg_sentiment.index,
                    title='Overall Sentiment Distribution')
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Speaking Time Analysis")
        # Speaking time bar chart
        fig = px.bar(x=list(speaking_time.keys()),
                    y=list(speaking_time.values()),
                    title='Speaking Time per Speaker')
        st.plotly_chart(fig)
    
    with tab4:
        st.subheader("Action Items Analysis")
        # Action items timeline
        action_items_list = action_items.split('\n')
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Action Item', 'Assignee', 'Deadline'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[[item.split(' - ')[0] for item in action_items_list],
                             [item.split(' - ')[1] if ' - ' in item else 'Unassigned' for item in action_items_list],
                             [item.split(' - ')[2] if ' - ' in item else 'No deadline' for item in action_items_list]],
                      fill_color='lavender',
                      align='left'))
        ])
        st.plotly_chart(fig)

def create_calendar_event(summary, start_time, end_time, description):
    """Create a Google Calendar event for the meeting."""
    try:
        service = get_google_calendar_service()
        event = {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'UTC',
            },
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        return event.get('htmlLink')
    except Exception as e:
        st.error(f"Error creating calendar event: {str(e)}")
        return None

def convert_audio_to_wav(audio_file):
    """Convert uploaded audio file to WAV format."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
        temp_wav_path = temp_wav.name
    
    # Convert to WAV using pydub
    audio = AudioSegment.from_file(audio_file)
    audio.export(temp_wav_path, format="wav")
    return temp_wav_path

def transcribe_audio(audio_file):
    """Transcribe audio file to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            return None

def analyze_sentiment(text):
    """Analyze sentiment of the meeting transcript."""
    sia = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    sentiments = []
    
    for sentence in sentences:
        sentiment = sia.polarity_scores(sentence)
        sentiments.append({
            'sentence': sentence,
            'compound': sentiment['compound'],
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu']
        })
    
    return sentiments

def extract_action_items(text):
    """Extract action items from the transcript using Groq."""
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "Extract action items from the following meeting transcript. Format them as a list with assignees and deadlines if mentioned."},
                {"role": "user", "content": text}
            ],
            max_tokens=300,
            temperature=0.7,
            top_p=0.95
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error extracting action items: {str(e)}")
        return None

def summarize_text(text):
    """Summarize text using Groq's LLM."""
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",  # Using Mixtral model for better performance
            messages=[
                {"role": "system", "content": """You are a professional meeting summarizer. Create a comprehensive summary of the following meeting transcript. 
                Include:
                1. Key points discussed
                2. Decisions made
                3. Action items with assignees
                4. Timeline of important events
                5. Risk factors or concerns raised
                Format the summary in clear sections."""},
                {"role": "user", "content": text}
            ],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.95
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

def export_to_pdf(summary, action_items, sentiment_analysis):
    """Export meeting summary to PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Meeting Summary", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    
    # Add summary
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, summary)
    
    # Add action items
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Action Items", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, action_items)
    
    # Save PDF
    pdf_path = "meeting_summary.pdf"
    pdf.output(pdf_path)
    return pdf_path

def export_to_docx(summary, action_items, sentiment_analysis):
    """Export meeting summary to DOCX."""
    doc = Document()
    doc.add_heading("Meeting Summary", 0)
    
    # Add summary
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(summary)
    
    # Add action items
    doc.add_heading("Action Items", level=1)
    doc.add_paragraph(action_items)
    
    # Save DOCX
    docx_path = "meeting_summary.docx"
    doc.save(docx_path)
    return docx_path

def main():
    st.title("Advanced Meeting Summarizer AI")
    st.write("Upload your meeting audio/video file for comprehensive analysis and insights.")

    # Sidebar for additional options
    st.sidebar.header("Options")
    export_format = st.sidebar.selectbox(
        "Choose export format",
        ["PDF", "DOCX", "TXT"]
    )
    
    # Calendar integration option
    create_calendar = st.sidebar.checkbox("Create Calendar Event")
    if create_calendar:
        meeting_title = st.sidebar.text_input("Meeting Title")
        meeting_date = st.sidebar.date_input("Meeting Date")
        meeting_time = st.sidebar.time_input("Meeting Time")

    uploaded_file = st.file_uploader("Choose an audio/video file", 
                                   type=['mp3', 'wav', 'm4a', 'ogg', 'mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        with st.spinner("Processing your file..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            # Handle video files
            if uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
                temp_file_path = extract_audio_from_video(temp_file_path)

            # Convert to WAV if necessary
            if not uploaded_file.name.lower().endswith('.wav'):
                temp_file_path = convert_audio_to_wav(temp_file_path)

            # Reduce noise
            temp_file_path = reduce_noise(temp_file_path)

            # Transcribe audio
            st.write("Transcribing audio...")
            transcription = transcribe_audio(temp_file_path)
            
            if transcription:
                # Create tabs for different sections
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Transcription", "Summary", "Action Items", "Analysis", "Analytics Dashboard"
                ])
                
                with tab1:
                    st.subheader("Transcription")
                    st.write(transcription)

                # Generate enhanced summary
                st.write("Generating comprehensive summary...")
                summary = summarize_text(transcription)
                
                with tab2:
                    if summary:
                        st.subheader("Summary")
                        st.write(summary)

                # Extract action items
                action_items = extract_action_items(transcription)
                
                with tab3:
                    if action_items:
                        st.subheader("Action Items")
                        st.write(action_items)

                # Analyze sentiment
                sentiments = analyze_sentiment(transcription)
                
                with tab4:
                    st.subheader("Sentiment Analysis")
                    df = pd.DataFrame(sentiments)
                    fig = px.line(df, y=['positive', 'negative', 'neutral'], 
                                title='Sentiment Analysis Over Time')
                    st.plotly_chart(fig)

                # Create analytics dashboard
                with tab5:
                    create_analytics_dashboard(
                        transcription,
                        sentiments,
                        {"Speaker 1": [(0, 100), (200, 300)], "Speaker 2": [(100, 200), (300, 400)]},  # Example speaking time
                        action_items
                    )

                # Export functionality
                if st.sidebar.button("Export Summary"):
                    if export_format == "PDF":
                        file_path = export_to_pdf(summary, action_items, sentiments)
                    elif export_format == "DOCX":
                        file_path = export_to_docx(summary, action_items, sentiments)
                    else:  # TXT
                        with open("meeting_summary.txt", "w") as f:
                            f.write(f"Summary:\n{summary}\n\nAction Items:\n{action_items}")
                        file_path = "meeting_summary.txt"
                    
                    with open(file_path, "rb") as f:
                        st.sidebar.download_button(
                            label=f"Download {export_format}",
                            data=f,
                            file_name=f"meeting_summary.{export_format.lower()}",
                            mime=f"application/{export_format.lower()}"
                        )

                # Create calendar event if requested
                if create_calendar and meeting_title:
                    meeting_datetime = datetime.combine(meeting_date, meeting_time)
                    end_datetime = meeting_datetime.replace(hour=meeting_datetime.hour + 1)
                    calendar_link = create_calendar_event(
                        meeting_title,
                        meeting_datetime,
                        end_datetime,
                        f"Meeting Summary:\n{summary}\n\nAction Items:\n{action_items}"
                    )
                    if calendar_link:
                        st.sidebar.success(f"Calendar event created! [View Event]({calendar_link})")

            # Clean up temporary files
            os.unlink(temp_file_path)
            if temp_file_path != uploaded_file.name:
                os.unlink(temp_file_path)

if __name__ == "__main__":
    main() 