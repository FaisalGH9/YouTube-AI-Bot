import os
import subprocess
from pydub import AudioSegment
from typing import Optional, Callable, Any
from src.config.settings import DEFAULT_BITRATE, LONG_VIDEO_BITRATE, LONG_VIDEO_THRESHOLD_MINUTES

def get_file_size_mb(path: str) -> float:
    """Calculate file size in megabytes"""
    return os.path.getsize(path) / (1024 * 1024)

class AudioProcessor:
    """Handles audio processing operations"""
    
    @staticmethod
    def get_audio_duration_ms(audio_path: str) -> int:
        """
        Get the duration of an audio file in milliseconds
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in milliseconds
        """
        try:
            sound = AudioSegment.from_file(audio_path)
            return len(sound)
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            # Fallback using ffprobe
            try:
                import subprocess
                cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-show_entries", "format=duration", 
                    "-of", "default=noprint_wrappers=1:nokey=1", 
                    audio_path
                ]
                output = subprocess.check_output(cmd).decode('utf-8').strip()
                duration_seconds = float(output)
                return int(duration_seconds * 1000)
            except:
                # Return a default if everything fails
                return 0
    
    @staticmethod
    def compress_audio(input_path: str, output_path: str = "compressed_audio.mp3", 
                      override_bitrate: Optional[str] = None,
                      progress_callback: Optional[Callable[[str, float, str, Optional[float]], Any]] = None) -> str:
        """
        Enhanced audio compression with automatic bitrate selection based on video length
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save compressed audio
            override_bitrate: Optional manual bitrate setting
            progress_callback: Function to report progress
            
        Returns:
            Path to compressed audio file
        """
        try:
            sound = AudioSegment.from_file(input_path)
            
            # Determine video length in minutes
            duration_minutes = len(sound) / (1000 * 60)
            
            # Select bitrate based on duration unless manually overridden
            if override_bitrate:
                bitrate = override_bitrate
            else:
                if duration_minutes > LONG_VIDEO_THRESHOLD_MINUTES:
                    bitrate = LONG_VIDEO_BITRATE  # Use low bitrate for longer videos
                else:
                    bitrate = DEFAULT_BITRATE  # Use standard bitrate for shorter videos
            
            if progress_callback:
                progress_callback("Audio Processing", 10, 
                                 f"Video duration: {duration_minutes:.1f} minutes. Using {bitrate} bitrate.",
                                 duration_minutes * 1.5)
            
            # Convert to mono (often more reliable for transcription)
            sound = sound.set_channels(1)
            if progress_callback:
                progress_callback("Audio Processing", 30, "Converting to mono", duration_minutes * 1.0)
            
            # Set sample rate to 16kHz (OpenAI recommended)
            sound = sound.set_frame_rate(16000)
            if progress_callback:
                progress_callback("Audio Processing", 50, "Setting sample rate", duration_minutes * 0.7)
            
            # Export with compression
            if progress_callback:
                progress_callback("Audio Processing", 70, "Compressing audio", duration_minutes * 0.5)
                
            sound.export(output_path, format="mp3", bitrate=bitrate, 
                        parameters=["-ac", "1", "-ar", "16000"])
            
            if progress_callback:
                progress_callback("Audio Processing", 100, "Compression complete", 0)
            
            return output_path
        except Exception as e:
            print(f"Error during audio compression: {e}")
            # Fallback to direct ffmpeg conversion if pydub fails
            
            # Determine bitrate for fallback method
            if not override_bitrate:
                # We need to manually check duration for ffmpeg fallback
                try:
                    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                                  "-of", "default=noprint_wrappers=1:nokey=1", input_path]
                    duration_output = subprocess.check_output(duration_cmd).decode('utf-8').strip()
                    duration_minutes = float(duration_output) / 60
                    bitrate = LONG_VIDEO_BITRATE if duration_minutes > LONG_VIDEO_THRESHOLD_MINUTES else DEFAULT_BITRATE
                except:
                    # If ffprobe fails, use default bitrate
                    bitrate = DEFAULT_BITRATE
            else:
                bitrate = override_bitrate
                
            if progress_callback:
                progress_callback("Audio Processing", 80, "Using fallback compression method", None)
                
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path, 
                "-ac", "1", "-ar", "16000", "-b:a", bitrate, output_path
            ])
            
            if progress_callback:
                progress_callback("Audio Processing", 100, "Compression complete (fallback method)", 0)
                
            return output_path

    @staticmethod
    def trim_audio_to_size_limit(input_path: str, target_path: str = "clipped_audio.mp3", 
                                bitrate: str = "8k", max_size_mb: int = 25) -> str:
        """
        Trim audio to fit within size limit (e.g., for API limitations)
        
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
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Calculate duration limit based on bitrate
        bitrate_kbps = int(bitrate.replace("k", ""))
        seconds_limit = int((max_size_mb * 1024 * 1024 * 8) / (bitrate_kbps * 1024))
        
        # Clip the audio
        clipped = sound[:seconds_limit * 1000]
        clipped.export(target_path, format="mp3", bitrate=bitrate)
        
        return target_path

    @staticmethod
    def process_audio_for_duration(input_path: str, output_path: str = "clipped_audio.mp3",
                                 duration_choice: str = "Full video", bitrate: str = "16k") -> str:
        """
        Process audio according to selected duration
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            duration_choice: Duration option (e.g., "First 10 minutes")
            bitrate: Bitrate for compression
            
        Returns:
            Path to processed audio file
        """
        if duration_choice == "Full video":
            return input_path
            
        # Extract minutes from duration choice
        limit_minutes = int(duration_choice.split()[1])
        
        sound = AudioSegment.from_file(input_path)
        clipped = sound[:limit_minutes * 60 * 1000]
        clipped.export(output_path, format="mp3", bitrate=bitrate)
        
        return output_path

    @staticmethod
    def split_audio_into_segments(audio_path: str, segment_size_minutes: int = 15, 
                                output_dir: str = "./temp_segments") -> list:
        """
        Split audio into segments for processing
        
        Args:
            audio_path: Path to audio file
            segment_size_minutes: Size of each segment in minutes
            output_dir: Directory to save segments
            
        Returns:
            List of segment file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        sound = AudioSegment.from_file(audio_path)
        total_duration_ms = len(sound)
        segment_size_ms = segment_size_minutes * 60 * 1000
        
        segment_paths = []
        
        for i in range(0, total_duration_ms, segment_size_ms):
            start_ms = i
            end_ms = min(i + segment_size_ms, total_duration_ms)
            
            segment = sound[start_ms:end_ms]
            segment_path = os.path.join(output_dir, f"segment_{i // segment_size_ms}.mp3")
            segment.export(segment_path, format="mp3", bitrate="16k")
            
            segment_paths.append({
                "path": segment_path,
                "segment_num": i // segment_size_ms,
                "start_ms": start_ms,
                "end_ms": end_ms
            })
        
        return segment_paths

    @staticmethod
    def is_long_video(audio_path: str, threshold_minutes: int = 60) -> bool:
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