import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# File paths
FFMPEG_PATH = os.getenv('FFMPEG_PATH', 'ffmpeg')  # Default to system ffmpeg if not specified

# Cache and database settings
CACHE_DIR = os.getenv('CACHE_DIR', './cache')
DB_DIR = os.getenv('DB_DIR', './chroma_db')

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Processing settings
DEFAULT_BITRATE = "16k"
LONG_VIDEO_BITRATE = "8k"
LONG_VIDEO_THRESHOLD_MINUTES = 45  # Videos longer than this use low bitrate

# Parallel processing settings
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '3'))
SEGMENT_SIZE_MINUTES = int(os.getenv('SEGMENT_SIZE_MINUTES', '10'))

# Model settings
DEFAULT_QA_MODEL = "gpt-3.5-turbo-instruct"
DEFAULT_SUMMARY_MODEL = "gpt-3.5-turbo-instruct"
TRANSCRIPTION_MODEL = "whisper-1"