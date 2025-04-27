import streamlit as st
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any

@dataclass
class ProgressState:
    """State for tracking progress"""
    step: str
    percentage: float
    message: str
    remaining_seconds: Optional[float] = None
    start_time: float = time.time()

class ProgressManager:
    """Manages progress display and tracking"""
    
    def __init__(self):
        self.progress_bar = None
        self.status_container = None
        self.current_state = None
        self._start_time = time.time()
    
    def initialize(self):
        """Initialize UI elements"""
        self.progress_bar = st.progress(0)
        self.status_container = st.empty()
        return self
    
    def update(self, step: str, percentage: float, message: str = None, 
              remaining_seconds: float = None) -> None:
        """
        Update progress display
        
        Args:
            step: Current processing step name
            percentage: Completion percentage (0-100)
            message: Status message to display
            remaining_seconds: Estimated time remaining
        """
        # Create progress state
        self.current_state = ProgressState(
            step=step,
            percentage=percentage,
            message=message or "",
            remaining_seconds=remaining_seconds,
            start_time=time.time()
        )
        
        # Update progress bar (ensure percentage is valid)
        valid_percentage = max(0, min(100, percentage)) / 100
        if self.progress_bar:
            self.progress_bar.progress(valid_percentage)
        
        # Format status message
        status = f"**{step}:** {percentage:.1f}%"
        if message:
            status += f" - {message}"
        
        # Add time estimate if available
        if remaining_seconds is not None and remaining_seconds > 0:
            if remaining_seconds > 60:
                mins = int(remaining_seconds // 60)
                secs = int(remaining_seconds % 60)
                status += f" (Est. remaining: {mins}m {secs}s)"
            else:
                status += f" (Est. remaining: {int(remaining_seconds)}s)"
        
        # Update status display
        if self.status_container:
            self.status_container.markdown(status)
    
    def clear(self) -> None:
        """Clear progress display"""
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_container:
            self.status_container.empty()
        
    def get_current_state(self) -> Optional[ProgressState]:
        """Get current progress state"""
        return self.current_state