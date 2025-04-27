import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.utils.downloader import get_video_id, download_audio_from_youtube
from src.utils.audio_processing import (
    get_file_size_mb,
    compress_audio,
    trim_audio_to_size_limit
)
from src.utils.transcription import transcribe_audio_with_openai
from src.config.settings import OPENAI_API_KEY
from langchain.embeddings.openai import OpenAIEmbeddings
from pydub import AudioSegment

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def create_vector_db_from_youtube_url(video_url: str, duration_choice="Full video"):
    """
    Create a vector database from a YouTube video URL
    """
    video_id = get_video_id(video_url)
    db_path = f"./chroma_db/{video_id}"

    if os.path.exists(db_path):
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        return db, 0

    # Try downloading with enhanced error handling
    audio_path = download_audio_from_youtube(video_url)
    
    # Use better compression settings for OpenAI compatibility
    compressed_audio_path = compress_audio(audio_path, bitrate="8k")
    size_mb = get_file_size_mb(compressed_audio_path)

    if size_mb > 25:
        final_audio_path = trim_audio_to_size_limit(compressed_audio_path)
    elif duration_choice != "Full video":
        limit_minutes = int(duration_choice.split()[1])
        from pydub import AudioSegment
        sound = AudioSegment.from_file(compressed_audio_path)
        clipped = sound[:limit_minutes * 60 * 1000]
        clipped.export("clipped_audio.mp3", format="mp3", bitrate="8k")
        final_audio_path = "clipped_audio.mp3"
    else:
        final_audio_path = compressed_audio_path

    # Use enhanced transcription with error handling
    transcript_text = transcribe_audio_with_openai(final_audio_path)
    doc = Document(page_content=transcript_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    docs = text_splitter.split_documents([doc])

    db = Chroma.from_documents(docs, embeddings, persist_directory=db_path)
    db.persist()

    return db, size_mb

def create_vector_db_from_youtube_url_progressive(video_url: str, duration_choice="Full video", progress_callback=None):
    """
    Progressive version that processes the video in manageable chunks and 
    provides progress updates via callback.
    """
    video_id = get_video_id(video_url)
    db_path = f"./chroma_db/{video_id}"

    # Check for existing processed database
    if os.path.exists(db_path):
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        if progress_callback:
            progress_callback(1, 1, "Using existing database", db_path)
        return db, 0

    # Download and compress audio
    if progress_callback:
        progress_callback(0, 4, "Downloading audio", None)
    
    audio_path = download_audio_from_youtube(video_url)
    
    if progress_callback:
        progress_callback(1, 4, "Compressing audio", None)
    
    compressed_audio_path = compress_audio(audio_path, bitrate="6k")  # Reduced bitrate for long videos
    size_mb = get_file_size_mb(compressed_audio_path)

    # Handle duration choice
    if duration_choice != "Full video":
        if progress_callback:
            progress_callback(2, 4, f"Trimming to {duration_choice}", None)
        
        limit_minutes = int(duration_choice.split()[1])
        sound = AudioSegment.from_file(compressed_audio_path)
        clipped = sound[:limit_minutes * 60 * 1000]
        clipped.export("clipped_audio.mp3", format="mp3", bitrate="6k")
        final_audio_path = "clipped_audio.mp3"
    else:
        final_audio_path = compressed_audio_path

    # Process in segments for very long videos (> 60 minutes)
    sound = AudioSegment.from_file(final_audio_path)
    total_duration_minutes = len(sound) / (60 * 1000)
    
    # If video is longer than 60 minutes, process in 15-minute segments
    if total_duration_minutes > 60:
        return _process_long_video(sound, video_id, db_path, progress_callback)
    else:
        # For shorter videos, use the standard approach
        if progress_callback:
            progress_callback(3, 4, "Transcribing audio", None)
        
        transcript_text = transcribe_audio_with_openai(final_audio_path)
        
        if progress_callback:
            progress_callback(4, 4, "Creating vector database", None)
        
        # Use larger chunks with less overlap for efficiency
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
        doc = Document(page_content=transcript_text)
        docs = text_splitter.split_documents([doc])

        db = Chroma.from_documents(docs, embeddings, persist_directory=db_path)
        db.persist()

        return db, size_mb

def _process_long_video(sound, video_id, db_path, progress_callback=None):
    """Helper function to process long videos in segments."""
    # Set segment size to 15 minutes
    segment_length_ms = 15 * 60 * 1000
    total_segments = len(sound) // segment_length_ms + (1 if len(sound) % segment_length_ms > 0 else 0)
    
    # Create temporary directory for segments
    temp_dir = f"./temp_segments/{video_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    db = None
    combined_size_mb = 0
    
    for i in range(total_segments):
        # Extract and save segment
        start_ms = i * segment_length_ms
        end_ms = min((i + 1) * segment_length_ms, len(sound))
        
        if progress_callback:
            progress_callback(i, total_segments, 
                             f"Processing segment {i+1}/{total_segments} ({start_ms//60000}-{end_ms//60000} min)", 
                             None)
        
        segment = sound[start_ms:end_ms]
        segment_path = f"{temp_dir}/segment_{i}.mp3"
        segment.export(segment_path, format="mp3", bitrate="6k")
        
        # Get size
        segment_size_mb = get_file_size_mb(segment_path)
        combined_size_mb += segment_size_mb
        
        # Transcribe segment
        transcript_segment = transcribe_audio_with_openai(segment_path)
        
        # Process this segment into vector db
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=300)
        doc = Document(page_content=transcript_segment)
        docs = text_splitter.split_documents([doc])
        
        if db is None:
            # Initialize DB with first segment
            db = Chroma.from_documents(docs, embeddings, persist_directory=db_path)
        else:
            # Add documents to existing DB
            db.add_documents(docs)
            
        # Persist after each segment
        db.persist()
        
        # Clean up segment file
        os.remove(segment_path)
    
    # Clean up temp directory
    os.rmdir(temp_dir)
    
    return db, combined_size_mb