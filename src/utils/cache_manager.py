import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.config.settings import CACHE_DIR

class TranscriptionCache:
    """Manages caching of partial transcriptions for long videos"""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, video_id: str, segment_num: Optional[int] = None) -> str:
        """Get path for cache file"""
        if segment_num is not None:
            return os.path.join(self.cache_dir, f"{video_id}_segment_{segment_num}.json")
        else:
            return os.path.join(self.cache_dir, f"{video_id}_metadata.json")
    
    def save_segment(self, video_id: str, segment_num: int, 
                    start_time: int, end_time: int, transcript: str) -> str:
        """Save a transcribed segment to cache"""
        cache_file = self.get_cache_path(video_id, segment_num)
        
        data = {
            "segment": segment_num,
            "start_time": start_time,
            "end_time": end_time,
            "transcript": transcript,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        # Update metadata
        self._update_metadata(video_id, segment_num)
        
        return cache_file
    
    def _update_metadata(self, video_id: str, latest_segment: int) -> None:
        """Update metadata with latest processed segment info"""
        metadata_file = self.get_cache_path(video_id)
        
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"video_id": video_id, "segments": [], "last_updated": None}
            
            # Update segment info
            if latest_segment not in metadata["segments"]:
                metadata["segments"].append(latest_segment)
                metadata["segments"].sort()
            
            metadata["last_updated"] = datetime.now().isoformat()
            metadata["segments_completed"] = len(metadata["segments"])
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Error updating cache metadata: {e}")
    
    def get_cached_segments(self, video_id: str) -> List[Dict[str, Any]]:
        """Get all cached segments for a video"""
        metadata_file = self.get_cache_path(video_id)
        
        if not os.path.exists(metadata_file):
            return []
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        segments = []
        for segment_num in metadata.get("segments", []):
            segment_file = self.get_cache_path(video_id, segment_num)
            if os.path.exists(segment_file):
                with open(segment_file, 'r') as f:
                    segments.append(json.load(f))
        
        # Sort by segment number
        segments.sort(key=lambda x: x["segment"])
        return segments
    
    def combine_transcripts(self, video_id: str) -> Optional[str]:
        """Combine all cached segments into a full transcript"""
        segments = self.get_cached_segments(video_id)
        
        if not segments:
            return None
        
        # Combine all transcripts in order
        full_transcript = " ".join([s["transcript"] for s in segments])
        return full_transcript
    
    def is_fully_cached(self, video_id: str, total_segments: int) -> bool:
        """Check if all segments are cached"""
        metadata_file = self.get_cache_path(video_id)
        
        if not os.path.exists(metadata_file):
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        cached_segments = set(metadata.get("segments", []))
        all_segments = set(range(total_segments))
        
        return cached_segments == all_segments