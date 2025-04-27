import os
from pydub import AudioSegment

def get_file_size_mb(path):
    """Calculate file size in megabytes"""
    return os.path.getsize(path) / (1024 * 1024)

def compress_audio(input_path, output_path="compressed_audio.mp3", bitrate="16k"):
    """
    Enhanced audio compression with better OpenAI compatibility
    """
    try:
        sound = AudioSegment.from_file(input_path)
        
        # Convert to mono (often more reliable for transcription)
        sound = sound.set_channels(1)
        
        # Set sample rate to 16kHz (OpenAI recommended)
        sound = sound.set_frame_rate(16000)
        
        # Export in MP3 format with moderate compression
        sound.export(output_path, format="mp3", bitrate=bitrate, 
                    parameters=["-ac", "1", "-ar", "16000"])
        
        return output_path
    except Exception as e:
        print(f"Error during audio compression: {e}")
        # Fallback to direct ffmpeg conversion if pydub fails
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, 
            "-ac", "1", "-ar", "16000", "-b:a", bitrate, output_path
        ])
        return output_path

def trim_audio_to_size_limit(input_path, target_path="clipped_audio.mp3", bitrate="8k", max_size_mb=25):
    """
    Trim audio to fit within OpenAI's file size limit
    
    Args:
        input_path: Path to input audio file
        target_path: Path to save trimmed audio
        bitrate: Bitrate for compression
        max_size_mb: Maximum file size in MB
    
    Returns:
        Path to trimmed audio file
    """
    # Try more aggressive compression first
    sound = AudioSegment.from_file(input_path)
    
    # For very large files, try progressively more aggressive compression before trimming
    if get_file_size_mb(input_path) > max_size_mb * 2:
        # Try mono first
        sound = sound.set_channels(1)
        
        # Try reducing sample rate
        sound = sound.set_frame_rate(16000)
        
        # Try exporting with lower bitrate
        temp_path = "temp_compressed.mp3"
        sound.export(temp_path, format="mp3", bitrate="4k")
        
        # Check if compression was enough
        if get_file_size_mb(temp_path) <= max_size_mb:
            os.rename(temp_path, target_path)
            return target_path
        
        # If still too large, fall back to trimming
        os.remove(temp_path)
    
    # Calculate duration limit based on bitrate
    bitrate_kbps = int(bitrate.replace("k", ""))
    seconds_limit = int((max_size_mb * 1024 * 1024 * 8) / (bitrate_kbps * 1024))
    
    # Clip the audio
    clipped = sound[:seconds_limit * 1000]
    clipped.export(target_path, format="mp3", bitrate=bitrate)
    
    return target_path

def is_long_video(audio_path, threshold_minutes=60):
    """
    Check if audio duration exceeds threshold
    
    Args:
        audio_path: Path to audio file
        threshold_minutes: Threshold in minutes to consider "long"
        
    Returns:
        Boolean indicating if video is "long"
    """
    sound = AudioSegment.from_file(audio_path)
    duration_minutes = len(sound) / (1000 * 60)
    return duration_minutes > threshold_minutes