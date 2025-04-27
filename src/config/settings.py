import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

# OpenAI Model Settings
DEFAULT_QA_MODEL = os.getenv("DEFAULT_QA_MODEL", "gpt-3.5-turbo-instruct")
DEFAULT_SUMMARY_MODEL = os.getenv("DEFAULT_SUMMARY_MODEL", "gpt-3.5-turbo-instruct")
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "whisper-1")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002")

# Application Storage
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(BASE_DIR, "cache"))
DB_DIR = os.getenv("DB_DIR", os.path.join(BASE_DIR, "chroma_db"))

# Audio Processing
DEFAULT_BITRATE = os.getenv("DEFAULT_BITRATE", "32k")
LONG_VIDEO_BITRATE = os.getenv("LONG_VIDEO_BITRATE", "16k")
LONG_VIDEO_THRESHOLD_MINUTES = int(os.getenv("LONG_VIDEO_THRESHOLD_MINUTES", "60"))

# Performance Settings
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))

# External Tools
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# LangSmith Settings
LANGSMITH_PROJECT_NAME = os.getenv("LANGSMITH_PROJECT_NAME", "youtube-ai-assistant")
LANGSMITH_API_URL = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
LANGSMITH_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING_ENABLED", "false").lower() == "true"

# Create required directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)