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
    
    def download_video(self, youtube_url: str, output_path: str = None,
                    max_height: int = 720,
                    progress_callback: Optional[Callable[[str, float, str, Optional[float]], Any]] = None) -> str:
        """
        Download video from YouTube with enhanced error handling
        
        Args:
            youtube_url: YouTube video URL
            output_path: Base path for the output file (if None, use video ID)
            max_height: Maximum height of the video (for quality control)
            progress_callback: Function to report progress
            
        Returns:
            Path to downloaded video file
            
        Raises:
            RuntimeError: If download fails
            FileNotFoundError: If downloaded file is not found
        """
        # Generate video ID for consistent naming if no output path provided
        video_id = self.get_video_id(youtube_url)
        if output_path is None:
            output_path = f"video_{video_id}"
        
        # Choose format based on max_height
        format_str = f'bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best'
        
        ydl_opts = {
            'format': format_str,
            'outtmpl': f'{output_path}.%(ext)s',
            'ffmpeg-location': os.path.dirname(FFMPEG_PATH),
            'quiet': False,
            'retries': 10,
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            # Merge into mp4 container
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            # Don't download video over 2GB
            'max_filesize': 2 * 1024 * 1024 * 1024,  # 2GB
        }

        if progress_callback:
            progress_callback("Video Download", 10, "Initializing video download", None)
            
            # Add progress hooks
            ydl_opts['progress_hooks'] = [
                lambda d: self._video_progress_hook(d, progress_callback)
            ]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
        except Exception as e:
            # If standard method fails, try alternative formats
            if progress_callback:
                progress_callback("Video Download", 30, "Trying alternative format", None)
                
            # Try with a different format option (lower quality)
            ydl_opts['format'] = f'bestvideo[height<=480]+bestaudio/best[height<=480]/best'
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            except Exception as inner_e:
                # Try one last time with even simpler options
                try:
                    ydl_opts['format'] = 'worst'
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([youtube_url])
                except Exception as final_e:
                    raise RuntimeError(f"❌ Failed to download video. Reason: {final_e}")

        # Check for the output file with various possible extensions
        for ext in ['mp4', 'mkv', 'webm', 'avi']:
            final_video = f"{output_path}.{ext}"
            if os.path.exists(final_video):
                if progress_callback:
                    progress_callback("Video Download", 100, "Video download complete", 0)
                return final_video
        
        raise FileNotFoundError("❌ Video file not found after download. Check if video is private or restricted.")
            
    def _progress_hook(self, d: dict, progress_callback: Callable) -> None:
        """
        Process progress information from yt-dlp for audio
        
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
    
    def _video_progress_hook(self, d: dict, progress_callback: Callable) -> None:
        """
        Process progress information from yt-dlp for video
        
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
                
                # Scale to 10-90% range
                scaled_percent = 10 + (percent * 0.8)
                
                progress_callback("Video Download", scaled_percent, 
                                f"Downloading video: {d.get('_percent_str', '?%')} at {d.get('_speed_str', '?')}", 
                                eta)
            except:
                # Fallback if percentage calculation fails
                progress_callback("Video Download", 50, "Downloading video...", None)
                
        elif d['status'] == 'finished':
            progress_callback("Video Download", 90, "Download finished, processing video...", None)