import os
import asyncio
import aiofiles
from pydub import AudioSegment
import openai
from src.config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def split_audio_into_chunks(input_path, chunk_minutes=10, output_dir="chunks"):
    """
    Split an audio file into multiple chunks of fixed length (default: 10 min).
    Returns a list of chunk paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    sound = AudioSegment.from_file(input_path)
    chunk_length = chunk_minutes * 60 * 1000  # milliseconds

    chunks = []
    for i in range(0, len(sound), chunk_length):
        chunk = sound[i:i + chunk_length]
        chunk_path = os.path.join(output_dir, f"chunk_{i // chunk_length}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate="16k")
        chunks.append(chunk_path)

    return chunks


async def transcribe_chunk(path):
    """
    Asynchronously transcribe a single chunk using OpenAI Whisper API.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sync_whisper_transcribe, path)


def sync_whisper_transcribe(path):
    """
    Blocking call to Whisper API to transcribe a chunk (called via asyncio thread).
    """
    with open(path, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text
