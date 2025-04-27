import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Tuple, Optional, Callable, Any
from src.utils.downloader import VideoDownloader
from src.utils.audio_processor import AudioProcessor, get_file_size_mb
from src.utils.transcription import TranscriptionService
from src.config.settings import OPENAI_API_KEY, DB_DIR, MAX_CONCURRENT_REQUESTS
from langchain.embeddings.openai import OpenAIEmbeddings

class VectorStoreService:
    """Service for creating and managing vector stores from YouTube videos"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.downloader = VideoDownloader()
        self.audio_processor = AudioProcessor()
        self.transcription = TranscriptionService()
        
    def create_vector_db_from_youtube_url(self, video_url: str, duration_choice: str = "Full video", 
                                        progress_callback: Optional[Callable[[str, float, str, Optional[float]], Any]] = None,
                                        parallelization: int = MAX_CONCURRENT_REQUESTS) -> Tuple[Chroma, float]:
        """
        Create a vector database from a YouTube video URL
        
        Args:
            video_url: YouTube video URL
            duration_choice: How much of the video to process
            progress_callback: Function to report progress
            parallelization: Number of concurrent requests for processing
            
        Returns:
            Tuple of (vector database, audio size in MB)
        """
        # Get video ID for caching
        video_id = self.downloader.get_video_id(video_url)
        db_path = os.path.join(DB_DIR, video_id)

        # Check for existing processed database
        if os.path.exists(db_path):
            db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
            
            if progress_callback:
                progress_callback("Database", 100, "Using existing vector database", 0)
                
            return db, 0
        
        # Download audio with progress updates
        if progress_callback:
            progress_callback("Download", 0, "Starting download", None)
        
        audio_path = self.downloader.download_audio(
            video_url, 
            progress_callback=progress_callback
        )
        
        # Process audio with better progress updates
        if progress_callback:
            progress_callback("Processing", 0, "Preparing audio", None)
        
        compressed_audio_path = self.audio_processor.compress_audio(
            audio_path, 
            progress_callback=progress_callback
        )
        
        # Handle duration choice if not full video
        if duration_choice != "Full video":
            if progress_callback:
                progress_callback("Processing", 80, f"Trimming to {duration_choice}", None)
                
            processed_audio_path = self.audio_processor.process_audio_for_duration(
                compressed_audio_path,
                duration_choice=duration_choice
            )
        else:
            processed_audio_path = compressed_audio_path
        
        # Transcribe audio - pass parallelization setting to the transcription service
        if progress_callback:
            progress_callback("Transcription", 0, "Preparing transcription", None)
        
        # Set segment size based on video length
        segment_size_minutes = 5  # Default to 5-minute segments for parallelization
        sound_duration_ms = AudioProcessor.get_audio_duration_ms(processed_audio_path)
        sound_duration_minutes = sound_duration_ms / (60 * 1000)
        
        # For very long videos, use larger segments
        if sound_duration_minutes > 180:  # > 3 hours
            segment_size_minutes = 10
        
        # Custom function to pass parallelization to transcribe_audio_with_segments
        def transcribe_with_parallelization(audio_path, video_id, progress_callback):
            from src.utils.parallel_transcription import transcribe_with_parallelization as twp
            return twp(
                audio_path, 
                video_id,
                segment_size_minutes=segment_size_minutes,
                max_concurrent=parallelization,
                progress_callback=progress_callback
            )
        
        # For longer videos, use parallel transcription
        if sound_duration_minutes > 20:  # > 20 minutes
            full_transcript = transcribe_with_parallelization(
                processed_audio_path, 
                video_id, 
                progress_callback
            )
        else:
            # For shorter videos, use standard transcription
            full_transcript = self.transcription.transcribe_audio_with_segments(
                processed_audio_path,
                video_id,
                progress_callback=progress_callback
            )
        
        # Create vector database
        if progress_callback:
            progress_callback("Creating Database", 0, "Creating vector database", None)
        
        doc = Document(page_content=full_transcript)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        docs = text_splitter.split_documents([doc])
        
        total_docs = len(docs)
        for i, _ in enumerate(docs):
            db_progress = ((i+1) / total_docs) * 100
            if progress_callback:
                progress_callback("Creating Database", db_progress, 
                                f"Processing chunk {i+1}/{total_docs}", 
                                (total_docs - i - 1) * 2)  # 2 seconds per chunk estimate
        
        db = Chroma.from_documents(docs, self.embeddings, persist_directory=db_path)
        db.persist()
        
        if progress_callback:
            progress_callback("Complete", 100, "Processing complete", 0)
        
        return db, get_file_size_mb(compressed_audio_path)