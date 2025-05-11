# Advanced Meeting Summarizer AI

An intelligent meeting summarizer that transcribes audio/video from meetings and generates comprehensive summaries using AI, with advanced analytics and integration capabilities.

## Features

### Core Features
- Audio/video transcription from meetings
- AI-powered meeting summarization
- Action items extraction and tracking
- Sentiment analysis of meeting content
- Multiple export formats (PDF, DOCX, TXT)

### Advanced Audio Processing
- Support for multiple audio formats (MP3, WAV, M4A, OGG)
- Video file support (MP4, AVI, MOV)
- Background noise reduction
- Audio enhancement
- Automatic audio extraction from video files

### Analytics Dashboard
- Meeting Overview
  - Meeting duration statistics
  - Number of speakers
  - Action items count
- Sentiment Analysis
  - Real-time sentiment tracking
  - Sentiment distribution visualization
  - Positive/negative/neutral breakdown
- Speaking Time Analysis
  - Per-speaker speaking time
  - Speaking time distribution
  - Interactive charts
- Action Items Analysis
  - Task tracking
  - Assignee management
  - Deadline monitoring

### Integration Features
- Google Calendar Integration
  - Direct calendar event creation
  - Meeting summary inclusion
  - Customizable meeting duration
  - Calendar event links

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg (required for video processing):
   - Windows: Download from https://ffmpeg.org/download.html
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

4. Set up Google Calendar Integration:
   - Create a Google Cloud Project
   - Enable the Google Calendar API
   - Download credentials.json
   - Place it in the project root directory

5. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

6. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Basic Usage
1. Open the web interface in your browser
2. Upload an audio/video file of your meeting
3. Wait for the transcription and analysis
4. View the comprehensive summary and analytics

### Advanced Features

#### Analytics Dashboard
- Navigate to the "Analytics Dashboard" tab to view:
  - Meeting overview metrics
  - Sentiment analysis charts
  - Speaking time distribution
  - Action items tracking

#### Calendar Integration
1. Check "Create Calendar Event" in the sidebar
2. Enter meeting details:
   - Meeting title
   - Date
   - Time
3. The meeting summary and action items will be included in the calendar event

#### Export Options
1. Choose export format from the sidebar:
   - PDF: Professional document with formatting
   - DOCX: Editable Word document
   - TXT: Plain text format
2. Click "Export Summary" to download

### File Support
- Audio Formats: MP3, WAV, M4A, OGG
- Video Formats: MP4, AVI, MOV
- Maximum file size: 100MB

## Technical Details

### Audio Processing
- Automatic format conversion to WAV
- Background noise reduction using noisereduce
- Audio enhancement for better transcription
- Video to audio extraction

### Analytics
- Real-time sentiment analysis using NLTK
- Interactive visualizations with Plotly
- Comprehensive metrics calculation
- Action items parsing and tracking

### Integration
- Google Calendar API integration
- OAuth2 authentication
- Secure credential management
- Event creation and management

## Requirements

- Python 3.8 or higher
- OpenAI API key
- Google Cloud Project (for Calendar integration)
- FFmpeg (for video processing)
- Internet connection for API calls
- Sufficient disk space for temporary files

## Security Notes

- API keys are stored in environment variables
- Google credentials are stored securely
- Temporary files are automatically cleaned up
- No data is stored permanently

## Troubleshooting

### Common Issues
1. Audio Processing
   - Ensure FFmpeg is installed correctly
   - Check file format compatibility
   - Verify file size limits

2. Calendar Integration
   - Verify Google Cloud Project setup
   - Check credentials.json placement
   - Ensure Calendar API is enabled

3. API Issues
   - Verify API key validity
   - Check internet connection
   - Monitor API usage limits

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Deployment Options

### 1. Streamlit Cloud (Recommended for Quick Deployment)
1. Create a GitHub repository and push your code
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign up/Login with your GitHub account
4. Click "New app"
5. Select your repository and main file (app.py)
6. Add your environment variables:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
7. Deploy!

### 2. Heroku Deployment
1. Install Heroku CLI
2. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port $PORT
   ```
3. Create `runtime.txt`:
   ```
   python-3.9.16
   ```
4. Initialize Git and Heroku:
   ```bash
   git init
   heroku create your-app-name
   ```
5. Set environment variables:
   ```bash
   heroku config:set OPENAI_API_KEY=your_api_key_here
   ```
6. Deploy:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

### 3. AWS Deployment
1. Create an EC2 instance (t2.micro or larger)
2. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip ffmpeg
   pip3 install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
4. Run with nohup:
   ```bash
   nohup streamlit run app.py --server.port 8501 &
   ```
5. Configure security groups to allow port 8501

### 4. Docker Deployment
1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   RUN apt-get update && apt-get install -y \
       ffmpeg \
       && rm -rf /var/lib/apt/lists/*

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Build and run:
   ```bash
   docker build -t meeting-summarizer .
   docker run -p 8501:8501 -e OPENAI_API_KEY=your_api_key_here meeting-summarizer
   ```

### 5. Google Cloud Run
1. Install Google Cloud SDK
2. Build and push Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/meeting-summarizer
   ```
3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy meeting-summarizer \
     --image gcr.io/YOUR_PROJECT_ID/meeting-summarizer \
     --platform managed \
     --allow-unauthenticated \
     --set-env-vars="OPENAI_API_KEY=your_api_key_here"
   ```

## Deployment Considerations

### Environment Variables
- Keep API keys secure
- Use environment variables for all sensitive data
- Never commit credentials to version control

### Resource Requirements
- Minimum 1GB RAM
- 1 CPU core
- 10GB storage
- FFmpeg for audio processing

### Scaling Considerations
- Use load balancing for high traffic
- Implement caching for API responses
- Monitor API usage limits
- Set up auto-scaling for cloud deployments

### Security Best Practices
1. Enable HTTPS
2. Implement rate limiting
3. Set up proper authentication
4. Regular security updates
5. Monitor for suspicious activity

### Monitoring and Maintenance
1. Set up logging
2. Monitor API usage
3. Track error rates
4. Regular backups
5. Performance monitoring

## Cost Considerations

### Free Tier Options
- Streamlit Cloud: Free tier available
- Heroku: Free tier available (with limitations)
- AWS: Free tier for 12 months
- Google Cloud: Free tier available

### Paid Options
- Streamlit Cloud: $10/month for teams
- Heroku: $7/month for hobby dyno
- AWS: Pay-as-you-go pricing
- Google Cloud: Pay-as-you-go pricing

## Performance Optimization

### For Production Deployment
1. Implement caching
2. Use CDN for static files
3. Optimize audio processing
4. Set up proper error handling
5. Implement retry mechanisms

### Monitoring Tools
1. Application logs
2. API usage metrics
3. Error tracking
4. Performance metrics
5. User analytics

## Backup and Recovery

### Regular Backups
1. Database backups (if implemented)
2. Configuration backups
3. User data backups
4. Log backups

### Recovery Procedures
1. Document recovery steps
2. Test recovery procedures
3. Maintain backup schedules
4. Monitor backup success 