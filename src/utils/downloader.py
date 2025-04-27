import os
import hashlib
import yt_dlp
from typing import Optional, Callable, Any
from src.config.settings import FFMPEG_PATH

class VideoDownloader:
    """Handles downloading of videos from YouTube"""
    
    @staticmethod
    def get_video_id(url: str) -> str:
        """
        Get a unique identifier for a YouTube video URL
        
        Args:
            url: YouTube URL
            
        Returns:
            Hash string that uniquely identifies the video
        """
        return hashlib.md5(url.encode()).hexdigest()
    
    def download_audio(self, youtube_url: str, output_path: str = "audio",
                     progress_callback: Optional[Callable[[str, float, str, Optional[float]], Any]] = None) -> str:
        """
        Download audio from a YouTube video with enhanced error handling
        
        Args:
            youtube_url: YouTube video URL
            output_path: Base path for the output file
            progress_callback: Function to report progress
            
        Returns:
            Path to downloaded audio file
            
        Raises:
            RuntimeError: If download fails
            FileNotFoundError: If downloaded file is not found
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_path}.%(ext)s',
            'ffmpeg-location': os.path.dirname(FFMPEG_PATH),
            'quiet': False,  # Set to False to see detailed logs
            'retries': 10,  # Add retry attempts
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        if progress_callback:
            progress_callback("Download", 10, "Initializing download", None)
            
            # Add progress hooks
            ydl_opts['progress_hooks'] = [
                lambda d: self._progress_hook(d, progress_callback)
            ]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
        except Exception as e:
            # If standard method fails, try alternative formats
            if progress_callback:
                progress_callback("Download", 30, "Trying alternative format", None)
                
            # Try with a different format option
            ydl_opts['format'] = 'worstaudio/worst'  # Try with lowest quality to ensure it downloads
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            except Exception as inner_e:
                raise RuntimeError(f"❌ Failed to download audio. Reason: {inner_e}")

        final_mp3 = f"{output_path}.mp3"
        if not os.path.exists(final_mp3):
            raise FileNotFoundError("❌ Audio file not found after download. Check if video is private or restricted.")

        # Verify the file is valid
        try:
            from pydub import AudioSegment
            _ = AudioSegment.from_file(final_mp3)
            
            if progress_callback:
                progress_callback("Download", 100, "Download complete", 0)
                
            return final_mp3
        except Exception as e:
            raise RuntimeError(f"❌ Downloaded audio file appears to be corrupted. Reason: {e}")
            
    def _progress_hook(self, d: dict, progress_callback: Callable) -> None:
        """
        Process progress information from yt-dlp
        
        Args:
            d: Progress dictionary from yt-dlp
            progress_callback: Function to report progress
        """
        if d['status'] == 'downloading':
            try:
                # Calculate percentage
                percent = float(d.get('downloaded_bytes', 0)) / float(d.get('total_bytes', 100)) * 100
                
                # Calculate estimated time remaining
                eta = d.get('eta', None)
                
                # Update display with realistic percentage (download is part of the total process)
                # Scale to 10-90% range to better represent the download portion
                scaled_percent = 10 + (percent * 0.8)  # 10-90% range
                
                progress_callback("Download", scaled_percent, 
                                f"Downloading: {d.get('_percent_str', '?%')} at {d.get('_speed_str', '?')}", 
                                eta)
            except:
                # Fallback if percentage calculation fails
                progress_callback("Download", 50, "Downloading...", None)
                
        elif d['status'] == 'finished':
            progress_callback("Download", 90, "Download finished, processing audio...", None)