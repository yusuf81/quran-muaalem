"""Custom exceptions for Quran Muaalem application."""


class QuranMuaalemError(Exception):
    """Base exception for Quran Muaalem."""
    pass


class AudioProcessingError(QuranMuaalemError):
    """Error during audio processing."""
    pass


class ModelInferenceError(QuranMuaalemError):
    """Error during model inference."""
    pass


class SegmentationError(QuranMuaalemError):
    """Error during audio segmentation."""
    pass


class ModelLoadError(QuranMuaalemError):
    """Error loading ML models."""
    pass


class ValidationError(QuranMuaalemError):
    """Input validation error."""
    pass
