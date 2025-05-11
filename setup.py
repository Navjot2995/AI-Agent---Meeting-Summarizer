import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    """Download the required spaCy model"""
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

def main():
    """Main setup function"""
    try:
        # Create virtual environment if it doesn't exist
        if not os.path.exists('venv'):
            print("Creating virtual environment...")
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        
        # Activate virtual environment
        if sys.platform == "win32":
            activate_script = os.path.join("venv", "Scripts", "activate")
        else:
            activate_script = os.path.join("venv", "bin", "activate")
        
        # Install requirements
        install_requirements()
        
        # Download spaCy model
        download_spacy_model()
        
        # Download NLTK data
        download_nltk_data()
        
        print("\nSetup completed successfully!")
        print("\nTo run the application:")
        print("1. Activate the virtual environment:")
        if sys.platform == "win32":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("2. Run the application:")
        print("   streamlit run app.py")
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 