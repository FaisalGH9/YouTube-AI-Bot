from typing import Tuple, Optional, Callable, Any
from src.langchain_pipeline.vector_store import VectorStoreService
from src.langchain_pipeline.qa_chain import QuestionAnswerer
from src.langchain_pipeline.summarizer import VideoSummarizer
from src.config.settings import MAX_CONCURRENT_REQUESTS

class VideoProcessor:
    """Main processing pipeline for YouTube videos"""
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.qa_service = QuestionAnswerer()
        self.summarizer = VideoSummarizer()
        self.parallelization = MAX_CONCURRENT_REQUESTS
    
    def set_parallelization(self, value: int) -> None:
        """Set the level of parallelization for processing"""
        self.parallelization = max(1, min(5, value))  # Ensure it's between 1 and 5
    
    def process_video(self, video_url: str, duration_choice: str = "Full video", 
                     progress_callback: Optional[Callable[[str, float, str, Optional[float]], Any]] = None) -> Tuple:
        """
        Process a YouTube video and create a searchable database
        
        Args:
            video_url: YouTube video URL
            duration_choice: How much of the video to process
            progress_callback: Function to report progress
            
        Returns:
            Tuple of (vector database, audio size in MB)
        """
        # Pass the parallelization setting to the vector store service
        return self.vector_store.create_vector_db_from_youtube_url(
            video_url, 
            duration_choice=duration_choice,
            progress_callback=progress_callback,
            parallelization=self.parallelization
        )
    
    def answer_question(self, db, query: str, k: int = 4, model_name: str = None):
        """
        Answer a question about the video
        
        Args:
            db: Vector database
            query: User's question
            k: Number of chunks to use for context
            model_name: LLM model to use
            
        Returns:
            Tuple of (answer, relevant document chunks)
        """
        try:
            return self.qa_service.answer_question(db, query, k=k, model_name=model_name)
        except Exception as e:
            # Fallback to simpler method if standard method fails
            print(f"Standard QA failed: {e}, falling back to simple answer")
            return self.qa_service.simple_answer(db, query, model_name=model_name)
    
    def summarize_video(self, db, model_name: str = None, summary_length: str = "Moderate"):
        """
        Generate a summary of the video
        
        Args:
            db: Vector database
            model_name: LLM model to use
            summary_length: Desired length of summary
            
        Returns:
            Summary text
        """
        return self.summarizer.summarize(db, model_name=model_name, summary_length=summary_length)