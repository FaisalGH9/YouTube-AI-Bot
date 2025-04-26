import os
import openai
from src.config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def transcribe_audio_with_openai(audio_file_path):
    """
    Transcribe audio with comprehensive error handling and fallback strategies
    
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
                model="whisper-1",  # Use whisper-1 explicitly
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        # If standard method fails, try re-encoding the file
        print(f"Initial transcription failed: {e}. Trying with re-encoded audio...")
        
        # Re-encode with different parameters
        temp_output = "temp_fixed_audio.mp3"
        try:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_file_path, 
                "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", 
                "-b:a", "64k", temp_output
            ], check=True)
            
            if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                # Try again with re-encoded file
                with open(temp_output, "rb") as audio_file:
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                
                # Clean up temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
                return transcript.text
            else:
                raise RuntimeError(f"Re-encoding failed to produce a valid audio file")
                
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
                        model="whisper-1",
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