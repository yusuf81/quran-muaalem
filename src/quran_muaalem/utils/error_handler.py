"""Error handling utilities for user-friendly error messages."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_error_response(
    title: str,
    details: str = "",
    suggestion: str = "",
    error_type: str = "error"
) -> str:
    """Create user-friendly error HTML.
    
    Args:
        title: Main error title
        details: Optional detailed error description
        suggestion: Optional suggestion for user action
        error_type: Type of error - 'error', 'warning', or 'info'
        
    Returns:
        HTML string with formatted error message
    """
    colors = {
        "error": {"bg": "#fee2e2", "border": "#dc2626", "text": "#991b1b"},
        "warning": {"bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e"},
        "info": {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e40af"}
    }
    
    color_scheme = colors.get(error_type, colors["error"])
    
    icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[error_type]
    
    html = f"""
    <div style='padding: 15px; background-color: {color_scheme["bg"]}; 
                border-left: 4px solid {color_scheme["border"]}; 
                border-radius: 5px; margin: 10px 0;'>
        <h4 style='color: {color_scheme["text"]}; margin-top: 0;'>
            {icon} {title}
        </h4>
    """
    
    if details:
        html += f"<p style='color: {color_scheme['text']};'>{details}</p>"
    
    if suggestion:
        html += f"""
        <p style='color: {color_scheme['text']}; margin-bottom: 0;'>
            <strong>💡 Saran:</strong> {suggestion}
        </p>
        """
    
    html += "</div>"
    
    return html


def log_and_return_error(
    error: Exception,
    title: str,
    suggestion: str = "",
    log_level: str = "error"
) -> str:
    """Log error and return user-friendly message.
    
    Args:
        error: The exception that occurred
        title: User-facing error title
        suggestion: Optional suggestion for the user
        log_level: Logging level - 'error', 'warning', 'info'
        
    Returns:
        HTML error message for display
    """
    # Log the error with full traceback
    log_func = getattr(logger, log_level)
    log_func(f"{title}: {str(error)}", exc_info=True)
    
    # Return user-friendly message
    return create_error_response(
        title=title,
        details=str(error),
        suggestion=suggestion,
        error_type=log_level
    )
