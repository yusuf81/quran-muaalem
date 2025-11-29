"""Audio processing service for Quran Muaalem."""

import numpy as np
import matplotlib.pyplot as plt
import librosa
from typing import Tuple, Optional

from ..config import sampling_rate


def plot_waveform(wave: np.ndarray, sr: int, title: str):
    """Return a matplotlib figure for waveform.
    
    Args:
        wave: Audio waveform data
        sr: Sampling rate
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    times = np.arange(len(wave)) / sr
    ax.plot(times, wave, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def preprocess_waveform(
    wave: np.ndarray,
    sr: int,
    enable_preprocess: bool,
    enable_debug: bool
) -> Tuple[np.ndarray, Optional[Tuple]]:
    """Preprocess audio waveform by trimming silence and adding padding.
    
    Args:
        wave: Input audio waveform
        sr: Sampling rate
        enable_preprocess: Whether to enable preprocessing
        enable_debug: Whether to return debug plots
        
    Returns:
        Tuple of (processed_wave, debug_plots)
        debug_plots is either None, a single plot, or tuple of (before, after) plots
    """
    if not enable_preprocess:
        return wave, plot_waveform(wave, sr, "Original Waveform") if enable_debug else None

    # Trim silence
    trimmed_wave, _ = librosa.effects.trim(
        wave, top_db=25, frame_length=2048, hop_length=512
    )

    # Add padding (1 second on each side)
    padding = sr  # 1 second worth of samples
    trimmed_wave = np.pad(trimmed_wave, (padding, padding), mode="constant")

    # Return with optional debug plots
    if enable_debug:
        before_fig = plot_waveform(wave, sr, "Waveform asli")
        after_fig = plot_waveform(trimmed_wave, sr, "Waveform setelah pemrosesan")
        return trimmed_wave, (before_fig, after_fig)
    
    return trimmed_wave, plot_waveform(trimmed_wave, sr, "Waveform diproses")
