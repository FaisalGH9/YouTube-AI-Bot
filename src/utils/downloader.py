import os
import hashlib
import yt_dlp
from src.config.settings import FFMPEG_PATH

def get_video_id(url):
    """Get a unique identifier for a YouTube video URL"""
    return hashlib.md5(url.encode()).hexdigest()

def download_audio_from_youtube(youtube_url, output_path="audio"):
    """
    Download audio from a YouTube video with enhanced error handling
    and captcha prevention
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}.%(ext)s',
        'ffmpeg-location': os.path.dirname(FFMPEG_PATH),
        'quiet': True,
        'cookies-from-browser': 'chrome',  # Use your browser - 'firefox', 'edge', 'safari', etc.
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        # If the browser cookies method fails, try without it
        if "cookies-from-browser" in str(e):
            print("Error with browser cookies, trying without...")
            ydl_opts.pop('cookies-from-browser', None)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            except Exception as inner_e:
                raise RuntimeError(f"❌ Failed to download audio. Reason: {inner_e}")
        else:
            raise RuntimeError(f"❌ Failed to download audio. Reason: {e}")

    final_mp3 = f"{output_path}.mp3"
    if not os.path.exists(final_mp3):
        raise FileNotFoundError("❌ Audio file not found after download. Check if video is private or restricted.")

    # Verify the file is valid
    try:
        from pydub import AudioSegment
        _ = AudioSegment.from_file(final_mp3)
    except Exception as e:
        raise RuntimeError(f"❌ Downloaded audio file appears to be corrupted. Reason: {e}")

    return final_mp3