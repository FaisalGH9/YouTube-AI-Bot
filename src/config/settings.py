import os
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables from .env
load_dotenv()

# API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FFmpeg binary location
FFMPEG_PATH = r"C:\Users\HUAWEI\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.converter = FFMPEG_PATH
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)
