"""Analysis service for orchestrating Quran recitation analysis."""

from typing import Tuple, Optional
import numpy as np
from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from quran_transcript.utils import PartOfUthmaniWord
from librosa.core import load

from ..config import sampling_rate
from ..services.model_manager import ModelManager
from ..services.audio_service import preprocess_waveform
from ..explain_gradio import explain_for_gradio
from ..exceptions import (
    AudioProcessingError,
    ModelInferenceError,
    ValidationError,
    ModelLoadError,
)
from ..utils.error_handler import log_and_return_error


def process_single_verse(
    audio_path: str,
    sura_idx: int,
    aya_idx: int,
    enable_preprocess: bool,
    enable_debug: bool,
    moshaf: MoshafAttributes,
    model_manager: ModelManager,
) -> Tuple[Optional[any], str]:
    """Process single verse audio for analysis.
    
    Args:
        audio_path: Path to audio file
        sura_idx: Surah index
        aya_idx: Ayah index
        enable_preprocess: Whether to enable audio preprocessing
        enable_debug: Whether to show debug waveforms
        moshaf: MoshafAttributes for phonetizer
        model_manager: ModelManager instance
        
    Returns:
        Tuple of (debug_plot, explanation_html)
    """
    if audio_path is None:
        from ..utils.error_handler import create_error_response
        return (
            None,
            create_error_response(
                "Audio Tidak Ditemukan",
                suggestion="Silakan unggah file audio atau rekam audio terlebih dahulu"
            ),
        )

    try:
        # Validate inputs
        if not sura_idx or not aya_idx:
            raise ValidationError("Surah dan ayat harus dipilih")

        # Get Uthmani reference text
        try:
            uthmani_ref = Aya(int(sura_idx), int(aya_idx)).get().uthmani
        except PartOfUthmaniWord as e:
            raise ValidationError(
                f"Pilihan ayat tidak valid: {str(e)}. "
                "Pastikan Anda memilih kata lengkap, bukan sebagian kata."
            )

        phonetizer_out = quran_phonetizer(
            uthmani_ref, moshaf, remove_spaces=True
        )

        # Process audio
        try:
            wave, _ = load(audio_path, sr=sampling_rate, mono=True)
        except Exception as e:
            raise AudioProcessingError(
                f"Gagal memuat file audio: {str(e)}. "
                "Pastikan file dalam format yang didukung (WAV, MP3, dll.)"
            )

        processed_wave, debug_figs = preprocess_waveform(
            wave, sampling_rate, enable_preprocess, enable_debug
        )

        # Model inference
        try:
            outs = model_manager.muaalem([processed_wave], [phonetizer_out], sampling_rate=sampling_rate)
        except ModelLoadError as e:
            raise ModelInferenceError(f"Gagal memuat model: {str(e)}")
        except Exception as e:
            raise ModelInferenceError(
                f"Model gagal menganalisis audio: {str(e)}"
            )

        # Add explanation
        explanation_html = explain_for_gradio(
            outs[0].phonemes.text,
            phonetizer_out.phonemes,
            outs[0].sifat,
            phonetizer_out.sifat,
            uthmani_ref,
        )

        # Decide waveform outputs for debug
        if enable_debug and isinstance(debug_figs, tuple):
            before_fig, after_fig = debug_figs
            debug_plot = after_fig
        elif enable_debug and debug_figs is not None:
            debug_plot = debug_figs
        else:
            debug_plot = None

        return debug_plot, explanation_html

    except ValidationError as e:
        return (
            None,
            log_and_return_error(
                e, 
                "Validasi Gagal",
                suggestion="Periksa kembali pilihan surah dan ayat Anda",
                log_level="warning"
            )
        )
    
    except AudioProcessingError as e:
        return (
            None,
            log_and_return_error(
                e,
                "Gagal Memproses Audio",
                suggestion="Pastikan file audio valid dan tidak corrupt"
            )
        )
    
    except ModelInferenceError as e:
        return (
            None,
            log_and_return_error(
                e,
                "Gagal Melakukan Analisis",
                suggestion="Coba dengan audio yang lebih jelas atau hubungi support"
            )
        )
    
    except Exception as e:
        return (
            None,
            log_and_return_error(
                e,
                "Terjadi Kesalahan Tidak Terduga",
                suggestion="Silakan coba lagi atau laporkan masalah ini"
            )
        )
