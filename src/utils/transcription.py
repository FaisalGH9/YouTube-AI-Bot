import os
import subprocess
import openai
from typing import Optional, Callable, Any, List, Dict
from pydub import AudioSegment
from src.config.settings import OPENAI_API_KEY, TRANSCRIPTION_MODEL
from src.utils.cache_manager import TranscriptionCache
from src.utils.parallel_transcription import transcribe_with_parallelization

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

class TranscriptionService:
    """Handles audio transcription using OpenAI's API"""
    
    def __init__(self):
        self.cache = TranscriptionCache()
    
    def transcribe_audio_with_openai(self, audio_file_path: str) -> str:
        """
        Transcribe an audio file using OpenAI's API
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            
        Returns:
            Transcribed text
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the file is too large
            RuntimeError: For other transcription errors
        """
        # Check if file exists
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        # Check file size (OpenAI has a 25MB limit)
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        if file_size_mb > 25:
            raise ValueError(f"Audio file is too large: {file_size_mb:.2f}MB (max 25MB)")
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model=TRANSCRIPTION_MODEL,
                    file=audio_file
                )
            return transcript.text
        except Exception as e:
            # If standard method fails, try re-encoding the file
            print(f"Initial transcription failed: {e}. Trying with re-encoded audio...")
            
            # Re-encode with different parameters
            temp_output = "temp_fixed_audio.mp3"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", audio_file_path, 
                    "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", 
                    "-b:a", "64k", temp_output
                ], check=True)
                
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                    # Try again with re-encoded file
                    with open(temp_output, "rb") as audio_file:
                        transcript = openai.audio.transcriptions.create(
                            model=TRANSCRIPTION_MODEL,
                            file=audio_file
                        )
                    
                    # Clean up temp file
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                        
                    return transcript.text
                else:
                    raise RuntimeError("Re-encoding failed to produce a valid audio file")
                    
            except Exception as re_encode_error:
                # If re-encoding fails, try one more approach with different format
                try:
                    temp_output2 = "temp_fixed_audio.wav"
                    subprocess.run([
                        "ffmpeg", "-y", "-i", audio_file_path, 
                        "-ac", "1", "-ar", "16000", temp_output2
                    ], check=True)
                    
                    with open(temp_output2, "rb") as audio_file:
                        transcript = openai.audio.transcriptions.create(
                            model=TRANSCRIPTION_MODEL,
                            file=audio_file
                        )
                    
                    # Clean up temp files
                    for temp_file in [temp_output, temp_output2]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            
                    return transcript.text
                    
                except Exception as final_error:
                    # Clean up any temp files
                    for temp_file in [temp_output, temp_output2]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            
                    raise RuntimeError(f"All transcription methods failed. Original error: {e}. Final error: {final_error}")
    
    def transcribe_audio_segment(self, audio_file_path: str, video_id: str, 
                               segment_num: int, start_time: int, end_time: int) -> str:
        """
        Transcribe a segment of audio with caching
        
        Args:
            audio_file_path: Path to the audio file
            video_id: Unique identifier for the video
            segment_num: Segment number for caching
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            Transcribed text for the segment
        """
        # Check if already cached
        cached_segments = self.cache.get_cached_segments(video_id)
        for segment in cached_segments:
            if segment["segment"] == segment_num:
                print(f"Using cached transcription for segment {segment_num}")
                return segment["transcript"]
        
        # If not cached, perform transcription
        try:
            transcript = self.transcribe_audio_with_openai(audio_file_path)
            
            # Cache the result
            self.cache.save_segment(
                video_id, 
                segment_num, 
                start_time, 
                end_time, 
                transcript
            )
            
            return transcript
        except Exception as e:
            # Handle errors but don't stop processing
            print(f"Error transcribing segment {segment_num}: {e}")
            return f"[Transcription failed for segment {segment_num}]"
    
    def transcribe_audio_with_segments(self, audio_path: str, video_id: str, 
                                     segment_size_minutes: int = 10,
                                     progress_callback: Optional[Callable[[str, float, str, Optional[float]], Any]] = None) -> str:
        """
        Transcribe audio file by splitting into manageable segments
        
        Args:
            audio_path: Path to the audio file
            video_id: Unique identifier for the video
            segment_size_minutes: Size of each segment in minutes
            progress_callback: Function to report progress
            
        Returns:
            Full transcription of the audio
        """
        # Check file size to determine if we should use parallel processing
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        duration_estimate_minutes = file_size_mb * 0.5  # Rough estimate: 2MB per minute
        
        # Use parallel processing for longer files (> 15 minutes estimated)
        if duration_estimate_minutes > 15:
            return transcribe_with_parallelization(
                audio_path,
                video_id,
                segment_size_minutes=segment_size_minutes,
                max_concurrent=3,  # Adjust based on your rate limit and budget
                progress_callback=progress_callback
            )
        
        # For shorter files, use the original sequential method
        # First check if we have a complete cached transcription
        existing_transcript = self.cache.combine_transcripts(video_id)
        if existing_transcript:
            if progress_callback:
                progress_callback("Transcription", 100, "Using cached transcription", 0)
            return existing_transcript
        
        # Load the audio file
        sound = AudioSegment.from_file(audio_path)
        total_duration_ms = len(sound)
        segment_size_ms = segment_size_minutes * 60 * 1000
        
        # Calculate total segments
        total_segments = (total_duration_ms + segment_size_ms - 1) // segment_size_ms
        
        # Check for partially cached segments
        cached_segments = self.cache.get_cached_segments(video_id)
        cached_segment_nums = {s["segment"] for s in cached_segments}
        
        # Create temporary directory for segments
        temp_dir = f"./temp_segments_{video_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process segments and build transcription
        all_transcripts = [""] * total_segments
        
        # First, populate with cached segments
        for segment in cached_segments:
            all_transcripts[segment["segment"]] = segment["transcript"]
        
        # Process remaining segments
        for i in range(total_segments):
            # Skip already cached segments
            if i in cached_segment_nums:
                if progress_callback:
                    segment_progress = ((i + 1) / total_segments) * 100
                    progress_callback("Transcription", segment_progress, 
                                    f"Using cached segment {i+1}/{total_segments}", None)
                continue
                
            # Extract segment times
            start_ms = i * segment_size_ms
            end_ms = min((i + 1) * segment_size_ms, total_duration_ms)
            
            # Update progress
            if progress_callback:
                segment_progress = ((i + 0.5) / total_segments) * 100
                est_remaining = (total_segments - i - 0.5) * 60  # Rough estimate: 1 min per segment
                progress_callback("Transcription", segment_progress, 
                                f"Processing segment {i+1}/{total_segments} ({start_ms//60000}-{end_ms//60000} min)", 
                                est_remaining)
            
            # Extract and save segment
            segment = sound[start_ms:end_ms]
            segment_path = os.path.join(temp_dir, f"segment_{i}.mp3")
            segment.export(segment_path, format="mp3", bitrate="16k")
            
            # Transcribe segment
            transcript = self.transcribe_audio_segment(segment_path, video_id, i, start_ms, end_ms)
            all_transcripts[i] = transcript
            
            # Clean up segment file
            if os.path.exists(segment_path):
                os.remove(segment_path)
                
            # Update progress
            if progress_callback:
                segment_progress = ((i + 1) / total_segments) * 100
                est_remaining = (total_segments - i - 1) * 60
                progress_callback("Transcription", segment_progress, 
                                f"Completed segment {i+1}/{total_segments}", 
                                est_remaining)
        
        # Clean up temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        # Combine all transcripts
        full_transcript = " ".join(all_transcripts)
        
        if progress_callback:
            progress_callback("Transcription", 100, "Transcription complete", 0)
            
        return full_transcript