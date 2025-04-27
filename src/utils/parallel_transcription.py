import asyncio
import aiohttp
import os
import time
from pydub import AudioSegment
import openai
from src.config.settings import OPENAI_API_KEY
from src.utils.cache_manager import TranscriptionCache

class ParallelTranscriber:
    def __init__(self, cache_dir="./cache"):
        self.cache = TranscriptionCache(cache_dir)
        self.api_key = OPENAI_API_KEY
        
    async def transcribe_chunk(self, session, chunk_path, video_id, chunk_id, start_ms, end_ms):
        """Transcribe a single audio chunk using OpenAI API"""
        # Check cache first
        cached_segments = self.cache.get_cached_segments(video_id)
        for segment in cached_segments:
            if segment["segment"] == chunk_id:
                print(f"Using cached transcription for chunk {chunk_id}")
                return segment["transcript"]
        
        # If not in cache, transcribe
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            with open(chunk_path, "rb") as audio_file:
                data = aiohttp.FormData()
                data.add_field("file", audio_file, filename=os.path.basename(chunk_path))
                data.add_field("model", "whisper-1")
                
                async with session.post("https://api.openai.com/v1/audio/transcriptions", 
                                      headers=headers, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        transcript = result.get("text", "")
                        
                        # Cache the result
                        self.cache.save_segment(video_id, chunk_id, start_ms, end_ms, transcript)
                        return transcript
                    else:
                        error_text = await response.text()
                        print(f"API error: {error_text}")
                        return f"[Error transcribing chunk {chunk_id}]"
        except Exception as e:
            print(f"Error in chunk {chunk_id}: {e}")
            return f"[Error processing chunk {chunk_id}]"
    
    async def transcribe_audio_parallel(self, audio_path, video_id, segment_size_minutes=10, 
                                      max_concurrent=3, progress_callback=None):
        """Transcribe audio in parallel chunks with rate limiting"""
        # Check if we already have a complete transcription
        existing_transcript = self.cache.combine_transcripts(video_id)
        if existing_transcript:
            if progress_callback:
                progress_callback("Transcription", 100, "Using cached transcription", 0)
            return existing_transcript
        
        # Process the audio
        sound = AudioSegment.from_file(audio_path)
        segment_size_ms = segment_size_minutes * 60 * 1000
        total_duration_ms = len(sound)
        
        # Calculate total segments
        total_segments = (total_duration_ms + segment_size_ms - 1) // segment_size_ms
        
        # Check for partially cached segments
        cached_segments = self.cache.get_cached_segments(video_id)
        cached_segment_nums = {s["segment"] for s in cached_segments}
        
        # Create temporary directory for chunks
        temp_dir = f"./temp_chunks_{video_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Prepare array for all transcripts
        all_transcripts = [""] * total_segments
        
        # Fill in cached transcripts
        for segment in cached_segments:
            all_transcripts[segment["segment"]] = segment["transcript"]
        
        # Prepare chunks for uncached segments
        chunk_tasks = []
        semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests
        
        async def process_with_semaphore(session, chunk_path, chunk_id, start_ms, end_ms):
            if progress_callback:
                progress_callback("Transcription", 
                                 ((chunk_id + 0.5) / total_segments) * 100,
                                 f"Processing chunk {chunk_id+1}/{total_segments}", 
                                 None)
            
            async with semaphore:
                transcript = await self.transcribe_chunk(session, chunk_path, video_id, chunk_id, start_ms, end_ms)
                all_transcripts[chunk_id] = transcript
                
                if progress_callback:
                    progress_callback("Transcription", 
                                    ((chunk_id + 1) / total_segments) * 100,
                                    f"Completed chunk {chunk_id+1}/{total_segments}", 
                                    None)
                
                # Clean up chunk file
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                    
                return transcript
        
        # Process uncached segments
        async with aiohttp.ClientSession() as session:
            for i in range(total_segments):
                # Skip already cached segments
                if i in cached_segment_nums:
                    if progress_callback:
                        progress_callback("Transcription", 
                                         ((i + 1) / total_segments) * 100,
                                         f"Using cached chunk {i+1}/{total_segments}", 
                                         None)
                    continue
                
                # Extract segment times
                start_ms = i * segment_size_ms
                end_ms = min((i + 1) * segment_size_ms, total_duration_ms)
                
                # Extract and save segment
                segment = sound[start_ms:end_ms]
                
                # Compress aggressively to reduce API costs
                segment = segment.set_channels(1)  # Convert to mono
                segment = segment.set_frame_rate(16000)  # Set to 16kHz
                
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp3")
                segment.export(chunk_path, format="mp3", bitrate="32k")
                
                # Add to tasks
                task = process_with_semaphore(session, chunk_path, i, start_ms, end_ms)
                chunk_tasks.append(task)
            
            # Execute all tasks
            if chunk_tasks:
                await asyncio.gather(*chunk_tasks)
        
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

# Helper function to run the async code
def transcribe_with_parallelization(audio_path, video_id, segment_size_minutes=10,
                                 max_concurrent=3, progress_callback=None):
    """Synchronous wrapper for the async transcription"""
    transcriber = ParallelTranscriber()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(
            transcriber.transcribe_audio_parallel(
                audio_path, video_id, segment_size_minutes, max_concurrent, progress_callback
            )
        )
    finally:
        loop.close()