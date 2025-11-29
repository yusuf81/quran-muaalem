import logging
import os
from dataclasses import asdict
import json
from typing import Literal, Optional, Any, get_origin, get_args

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from quran_transcript.utils import PartOfUthmaniWord
from quran_transcript.phonetics.moshaf_attributes import (
    get_arabic_attributes,
    get_arabic_name,
)
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa.core import load
from pydantic.fields import FieldInfo, PydanticUndefined
import torch
import gradio as gr
import torchaudio

# Monkey patch for recitations-segmenter compatibility with newer torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = list_audio_backends

from quran_muaalem.inference import Muaalem
from quran_muaalem.muaalem_typing import MuaalemOutput
from quran_muaalem.explain import explain_for_terminal
from quran_muaalem.explain_gradio import explain_for_gradio
from quran_muaalem.exceptions import (
    AudioProcessingError,
    ModelInferenceError,
    SegmentationError,
    ValidationError,
    ModelLoadError,
)
from quran_muaalem.utils.error_handler import create_error_response, log_and_return_error
from quran_muaalem.services.model_manager import ModelManager

# Recitations Segmenter Imports
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
from recitations_segmenter import segment_recitations, read_audio, clean_speech_intervals



# Import from new config module
from quran_muaalem.config import (
    REQUIRED_MOSHAF_FIELDS,
    device,
    sampling_rate,
    sura_idx_to_name,
    sura_to_aya_count,
    default_moshaf,
)
from quran_muaalem.services.audio_service import plot_waveform, preprocess_waveform
from quran_muaalem.ui.components import (
    get_field_name,
    create_gradio_input_for_field,
    update_aya_dropdown,
    update_uthmani_ref,
    update_uthmani_ref_html,
    update_multi_verse_uthmani_preview,
)

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model manager for lazy loading  
model_manager = ModelManager(
    device=device,
    muaalem_model_id="obadx/muaalem-model-v3_2",
    segmenter_model_id="obadx/recitation-segmenter-v2"
)






def process_audio(
    audio,
    sura_idx,
    aya_idx,
    enable_preprocess: bool,
    enable_debug: bool,
    moshaf: MoshafAttributes,
):
    """Process single verse audio with analysis.
    
    Args:
        audio: Audio file path
        sura_idx: Surah index
        aya_idx: Ayah index
        enable_preprocess: Whether to enable audio preprocessing
        enable_debug: Whether to show debug waveforms
        moshaf: MoshafAttributes for phonetizer
    """
    if audio is None:
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
            wave, _ = load(audio, sr=sampling_rate, mono=True)
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





def process_multi_verse_audio(
    audio,
    sura_idx,
    start_aya,
    end_aya,
    full_sura_toggle,
    enable_preprocess,
    moshaf: MoshafAttributes,
):
    """Process multi-verse audio with automatic segmentation.
    
    Args:
        audio: Audio file path
        sura_idx: Surah index
        start_aya: Start ayah number
        end_aya: End ayah number
        full_sura_toggle: Whether to analyze full surah
        enable_preprocess: Whether to enable preprocessing
        moshaf: MoshafAttributes for phonetizer
    """
    if audio is None:
        return create_error_response(
            "Audio Tidak Ditemukan",
            suggestion="Silakan unggah file audio untuk analisis multi-ayat"
        )

    try:
        # Load segmenter model
        try:
            segmenter_model, segmenter_processor = model_manager.segmenter
        except ModelLoadError as e:
            raise ModelInferenceError(f"Gagal memuat model segmentasi: {str(e)}")

        # Validate and determine verses to process
        if not sura_idx:
            raise ValidationError("Surah harus dipilih")

        if full_sura_toggle:
            start_aya = 1
            end_aya = sura_to_aya_count[int(sura_idx)]
        
        try:
            start_aya = int(start_aya)
            end_aya = int(end_aya)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Nomor ayat tidak valid: {str(e)}")

        if start_aya > end_aya:
            raise ValidationError(
                f"Ayat awal ({start_aya}) harus lebih kecil atau sama dengan ayat akhir ({end_aya})"
            )
        
        if start_aya < 1 or end_aya > sura_to_aya_count[int(sura_idx)]:
            raise ValidationError(
                f"Rentang ayat tidak valid. Surah {sura_idx_to_name[int(sura_idx)]} "
                f"memiliki {sura_to_aya_count[int(sura_idx)]} ayat"
            )

        # Process audio for segmentation
        try:
            wave = read_audio(audio)
        except Exception as e:
            raise AudioProcessingError(f"Gagal membaca file audio: {str(e)}")

        # Segment
        try:
            sampled_outputs = segment_recitations(
                [wave],
                segmenter_model,
                segmenter_processor,
                device=device,
                dtype=torch.bfloat16,
                batch_size=1,
            )
        except Exception as e:
            raise SegmentationError(f"Gagal melakukan segmentasi audio: {str(e)}")

        output = sampled_outputs[0]

        clean_out = clean_speech_intervals(
            output.speech_intervals,
            output.is_complete,
            min_silence_duration_ms=200,
            min_speech_duration_ms=500,
            pad_duration_ms=0,
            return_seconds=True,
        )

        segments = clean_out.clean_speech_intervals
        
        expected_verses_count = end_aya - start_aya + 1
        detected_segments_count = len(segments)

        html_output = f"<h3>Hasil Analisis (Surah {sura_idx_to_name[int(sura_idx)]}: {start_aya}-{end_aya})</h3>"
        html_output += f"<p>Jumlah ayat diharapkan: {expected_verses_count}, Jumlah segmen terdeteksi: {detected_segments_count}</p>"

        if detected_segments_count != expected_verses_count:
            html_output += create_error_response(
                "Peringatan: Ketidakcocokan Segmentasi",
                details=f"Jumlah segmen ({detected_segments_count}) tidak sesuai dengan jumlah ayat ({expected_verses_count})",
                suggestion="Pastikan Anda berhenti sejenak (waqf) di setiap akhir ayat agar sistem dapat memisahkan dengan benar",
                error_type="warning"
            )

        # Load original full audio for slicing
        try:
            full_wave, _ = load(audio, sr=sampling_rate, mono=True)
        except Exception as e:
            raise AudioProcessingError(f"Gagal memuat audio untuk processing: {str(e)}")

        # Process each segment
        for i in range(min(expected_verses_count, detected_segments_count)):
            current_aya_idx = start_aya + i
            segment = segments[i]
            start_sec, end_sec = segment[0], segment[1]

            # Slice audio with small padding
            pad_samples = int(0.5 * sampling_rate)
            start_sample = max(0, int(start_sec * sampling_rate) - pad_samples)
            end_sample = min(len(full_wave), int(end_sec * sampling_rate) + pad_samples)

            aya_wave = full_wave[start_sample:end_sample]

            try:
                uthmani_ref = Aya(int(sura_idx), int(current_aya_idx)).get().uthmani
                phonetizer_out = quran_phonetizer(
                    uthmani_ref, moshaf, remove_spaces=True
                )

                # Preprocess (trim silence)
                processed_wave, _ = preprocess_waveform(
                    aya_wave, sampling_rate, enable_preprocess, False
                )

                outs = model_manager.muaalem(
                    [processed_wave], [phonetizer_out], sampling_rate=sampling_rate
                )

                explanation = explain_for_gradio(
                    outs[0].phonemes.text,
                    phonetizer_out.phonemes,
                    outs[0].sifat,
                    phonetizer_out.sifat,
                    uthmani_ref,
                )

                html_output += f"<div style='border:1px solid #e5e7eb; margin-bottom:20px; padding:15px; border-radius:8px; background-color: white;'>"
                html_output += f"<h4 style='margin-top:0;'>Ayat {current_aya_idx} <span style='font-weight:normal; font-size:0.9em; color:#666'>(Detik: {start_sec:.2f} - {end_sec:.2f})</span></h4>"
                
                # Display Utsmani text for reading
                html_output += f"<div style='font-size: 28px; line-height: 1.7; text-align: center; direction: rtl; font-family: \"Scheherazade New\", \"Amiri\", \"Noto Naskh Arabic\", \"Traditional Arabic\", serif; padding: 15px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 15px;'>{uthmani_ref}</div>"
                
                html_output += explanation
                html_output += "</div>"

            except Exception as e:
                logger.error(f"Error processing Ayat {current_aya_idx}: {e}", exc_info=True)
                html_output += f"<div style='margin: 10px 0;'>{create_error_response(f'Gagal Memproses Ayat {current_aya_idx}', details=str(e), error_type='error')}</div>"
        
        return html_output

    except ValidationError as e:
        return log_and_return_error(
            e,
            "Validasi Gagal",
            suggestion="Periksa kembali pilihan surah dan rentang ayat",
            log_level="warning"
        )
    
    except AudioProcessingError as e:
        return log_and_return_error(
            e,
            "Gagal Memproses Audio",
            suggestion="Pastikan file audio valid dan tidak corrupt"
        )
    
    except SegmentationError as e:
        return log_and_return_error(
            e,
            "Gagal Melakukan Segmentasi",
            suggestion="Pastikan audio memiliki jeda yang jelas antar ayat. Coba rekam ulang dengan jeda yang lebih jelas di akhir setiap ayat"
        )
    
    except ModelInferenceError as e:
        return log_and_return_error(
            e,
            "Gagal Memuat atau Menjalankan Model",
            suggestion="Model mungkin belum ter-download lengkap. Coba restart aplikasi atau periksa koneksi internet"
        )
    
    except Exception as e:
        return log_and_return_error(
            e,
            "Terjadi Kesalahan Tidak Terduga",
            suggestion="Silakan coba lagi atau laporkan masalah ini"
        )


def update_moshaf_settings(*args):
    """Update the moshaf settings with values from the settings page.
    
    Returns:
        Tuple of (new_moshaf, status_message)
    """
    try:
        # Create a dictionary from the field names and values
        settings_dict = dict(zip(field_names, args))

        # Create a new MoshafAttributes object with the updated values
        new_moshaf = MoshafAttributes(**settings_dict)
        return new_moshaf, "✅ Pengaturan berhasil disimpan - Settings saved successfully!"
    except Exception as e:
        return default_moshaf, f"❌ Kesalahan saat menyimpan pengaturan - Error saving settings: {str(e)}"


def reset_settings():
    """Reset all settings to default values.
    
    Returns:
        Tuple of (default_moshaf, field_values..., status_message)
    """
    try:
        # Return default values for all fields
        default_values = [
            getattr(default_moshaf, field_name) for field_name in field_names
        ]
        return (
            default_moshaf,
            *default_values,
            "✅ Berhasil mengembalikan ke pengaturan awal - Reset to default settings successfully!"
        )
    except Exception as e:
        current_values = [getattr(default_moshaf, field_name) for field_name in field_names]
        return (
            default_moshaf,
            *current_values,
            f"❌ Error resetting settings: {str(e)}"
        )





# Create the Gradio app
with gr.Blocks(title="Pengajar Al-Quran") as app:
    # Store current moshaf settings in session state
    current_moshaf_state = gr.State(default_moshaf)

    # Initialize field names list
    field_names = []

    # Create sura dropdown with both index and name (used in main tab)
    sura_choices = [
        (f"{idx} - {sura_idx_to_name[idx]}", idx) for idx in range(1, 115)
    ]

    with gr.Tab("Analisis Bacaan"):
        gr.Markdown("# Analisis Bacaan Al-Quran")
        gr.Markdown(
            "Analisis bacaan untuk satu ayat atau beberapa ayat sekaligus. Pastikan Anda berhenti sejenak (waqf) di setiap akhir ayat agar sistem dapat memisahkan ayat dengan benar."
        )

        with gr.Row():
            with gr.Column(scale=1):
                mv_sura_dropdown = gr.Dropdown(
                    choices=sura_choices,
                    label="Surah",
                    value=1,
                    elem_id="mv_sura_dropdown",
                )
                
                with gr.Row():
                    mv_start_aya = gr.Number(label="Ayat Awal", value=1, precision=0)
                    mv_end_aya = gr.Number(label="Ayat Akhir", value=5, precision=0)
                
                mv_full_sura_toggle = gr.Checkbox(
                    label="Analisis Full Surah", value=False
                )
                
                mv_preprocess_checkbox = gr.Checkbox(
                    label="Aktifkan pemrosesan audio (trim keheningan per segmen)",
                    value=True,
                )

            with gr.Column(scale=2):
                mv_audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    label="Unggah atau Rekam Audio (Bacaan Panjang)",
                    type="filepath",
                    elem_id="mv_audio_input",
                )
                mv_analyze_btn = gr.Button(
                    "Mulai Analisis Multi-Ayat", variant="primary"
                )
                # Add preview of Quran text before analysis
                mv_uthmani_preview = gr.HTML(
                    label="Teks Rujukan Ayat (untuk dibaca)",
                    elem_id="mv_uthmani_preview",
                )
                mv_output_html = gr.HTML(
                    label="Hasil Pemeriksaan",
                    elem_id="mv_output_html",
                )
        
        # Initial load of preview
        app.load(
            update_multi_verse_uthmani_preview,
            inputs=[mv_sura_dropdown, mv_start_aya, mv_end_aya, mv_full_sura_toggle],
            outputs=[mv_uthmani_preview],
        )


        # Logic for full sura toggle
        def toggle_aya_inputs(is_full, sura_idx):
            if is_full:
                # When full sura is checked, set to 1 and total ayas, and disable
                total_ayas = sura_to_aya_count[int(sura_idx)]
                return {
                    mv_start_aya: gr.update(value=1, interactive=False),
                    mv_end_aya: gr.update(value=total_ayas, interactive=False),
                }
            else:
                # When unchecked, just enable the fields (keep current values)
                return {
                    mv_start_aya: gr.update(interactive=True),
                    mv_end_aya: gr.update(interactive=True),
                }

        mv_full_sura_toggle.change(
            toggle_aya_inputs,
            inputs=[mv_full_sura_toggle, mv_sura_dropdown],
            outputs=[mv_start_aya, mv_end_aya],
        ).then(
            update_multi_verse_uthmani_preview,
            inputs=[mv_sura_dropdown, mv_start_aya, mv_end_aya, mv_full_sura_toggle],
            outputs=[mv_uthmani_preview],
        )
        
        # Also update when sura changes and full sura is checked
        mv_sura_dropdown.change(
            toggle_aya_inputs,
            inputs=[mv_full_sura_toggle, mv_sura_dropdown],
            outputs=[mv_start_aya, mv_end_aya],
        ).then(
            update_multi_verse_uthmani_preview,
            inputs=[mv_sura_dropdown, mv_start_aya, mv_end_aya, mv_full_sura_toggle],
            outputs=[mv_uthmani_preview],
        )
        
        # Update preview when aya numbers change
        for component in [mv_start_aya, mv_end_aya]:
            component.change(
                update_multi_verse_uthmani_preview,
                inputs=[mv_sura_dropdown, mv_start_aya, mv_end_aya, mv_full_sura_toggle],
                outputs=[mv_uthmani_preview],
            )

        mv_analyze_btn.click(
            process_multi_verse_audio,
            inputs=[
                mv_audio_input,
                mv_sura_dropdown,
                mv_start_aya,
                mv_end_aya,
                mv_full_sura_toggle,
                mv_preprocess_checkbox,
                current_moshaf_state,  # Pass moshaf state
            ],
            outputs=[mv_output_html],
        )

    with gr.Tab("Pengaturan Mushaf - Moshaf Settings"):
        gr.Markdown("# Pengaturan Sifat Mushaf")
        gr.Markdown("Sesuaikan sifat mushaf sesuai dengan bacaan yang diinginkan")

        # Create settings inputs directly in the tab
        settings_components = []
        fields = MoshafAttributes.model_fields

        # Create inputs for all required fields
        for field_name in REQUIRED_MOSHAF_FIELDS:
            field_info = fields[field_name]
            input_component = create_gradio_input_for_field(
                field_name, field_info, getattr(default_moshaf, field_name, None)
            )
            settings_components.append(input_component)
            field_names.append(field_name)

        # Save button and status message
        with gr.Row():
            save_btn = gr.Button("Simpan Pengaturan - Save Settings", variant="primary")
            reset_btn = gr.Button("Kembalikan ke Awal - Reset to Default")

        status_message = gr.Markdown()

        # Save settings event
        save_btn.click(
            update_moshaf_settings, 
            inputs=settings_components, 
            outputs=[current_moshaf_state, status_message]  # Update state and show message
        )

        # Reset to default event
        reset_btn.click(
            reset_settings, 
            inputs=[], 
            outputs=[current_moshaf_state] + settings_components + [status_message]  # Update state, fields, and message
        )

    with gr.Tab("Tentang & Credit"):
        gr.Markdown("""
        # Tentang Aplikasi - About This Application
        
        Aplikasi ini menggunakan model **Muaalem** untuk deteksi kesalahan dan koreksi pengucapan dalam bacaan Al-Quran berdasarkan kaidah tajwid.
        
        *This application uses the **Muaalem** model for pronunciation error detection and correction in Quranic recitation based on tajweed rules.*
        
        ---
        
        ## 📄 Paper / Makalah Penelitian
        
        **Title:** Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning
        
        **Authors:** Abdullah Abdelfattah, Mahmoud I. Khalil, Hazem Abbas
        
        **Published:** arXiv preprint arXiv:2509.00094 (2025)
        
        **Links:**
        - 📖 [Read Paper on arXiv](https://arxiv.org/abs/2509.00094)
        - 🌐 [Project Website](https://obadx.github.io/prepare-quran-dataset/)
        - 📓 [PDF Version](https://arxiv.org/pdf/2509.00094)
        - 🔗 [DOI: 10.48550/arXiv.2509.00094](https://doi.org/10.48550/arXiv.2509.00094)
        
        ---
        
        ## 💻 Source Code / Kode Sumber
        
        **GitHub Repository:** [obadx/quran-muaalem](https://github.com/obadx/quran-muaalem)
        
        ---
        
        ## 📚 Citation / Sitasi
        
        Jika Anda menggunakan aplikasi ini atau model Muaalem dalam penelitian Anda, mohon sitasi paper berikut:
        
        *If you use this application or the Muaalem model in your research, please cite the following paper:*
        
        ### BibTeX
        
        ```bibtex
        @article{abdelfattah2025automatic,
          title={Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning},
          author={Abdelfattah, Abdullah and Khalil, Mahmoud I. and Abbas, Hazem},
          journal={arXiv preprint arXiv:2509.00094},
          year={2025},
          url={https://arxiv.org/abs/2509.00094},
          doi={10.48550/arXiv.2509.00094}
        }
        ```
        
        ### APA Style
        
        ```
        Abdelfattah, A., Khalil, M. I., & Abbas, H. (2025). Automatic Pronunciation Error 
        Detection and Correction of the Holy Quran's Learners Using Deep Learning. 
        arXiv preprint arXiv:2509.00094. https://doi.org/10.48550/arXiv.2509.00094
        ```

        
        ---
        
        ## 🔧 Modifications in This Version / Modifikasi dalam Versi Ini
        
        Aplikasi ini merupakan modifikasi dari repository asli dengan penambahan fitur-fitur berikut:
        
        *This application is a modified version of the original repository with the following additional features:*
        
        ### 1. 🌏 Indonesian Translation / Translasi Bahasa Indonesia
        - Interface bilingual (Indonesia & English) untuk kemudahan pengguna lokal
        - Bilingual interface (Indonesian & English) for local users' convenience
        
        ### 2. 📖 Automatic Verse Segmentation / Pemenggalan Ayat Otomatis
        - **Fitur analisis banyak ayat sekaligus** menggunakan model segmentation (`recitation-segmenter-v2`)
        - Otomatis memisahkan rekaman panjang menjadi ayat-per-ayat berdasarkan jeda (waqf)
        - Mendukung analisis full surah
        - **Multi-verse analysis feature** using segmentation model (`recitation-segmenter-v2`)
        - Automatically segments long recordings into individual verses based on pauses (waqf)
        - Supports full surah analysis
        
        ### 3. 🔇 Audio Preprocessing / Pengolahan Audio
        - Setiap segmen ayat dipotong dengan **padding 0.5 detik** di awal dan akhir untuk menjaga konteks
        - **Silence trimming** opsional dilakukan SETELAH pemenggalan untuk membersihkan keheningan
        - Each verse segment is sliced with **0.5-second padding** at the beginning and end to preserve context
        - Optional **silence trimming** is applied AFTER segmentation to remove silence
        
        ### 4. 📊 Enhanced UI / Peningkatan Antarmuka
        - Preview teks Al-Quran sebelum analisis
        - Hasil analisis per-ayat dengan timestamp
        - Quran text preview before analysis
        - Per-verse analysis results with timestamps
        
        **Modified by:** Yusuf (GitHub: [yusuf81](https://github.com/yusuf81))
        
        ---
        
        ## 🙏 Acknowledgments / Penghargaan
        
        Terima kasih kepada tim peneliti yang telah mengembangkan model Muaalem dan menyediakan dataset serta kode sumber secara open-source.
        
        *Thanks to the research team who developed the Muaalem model and provided the dataset and source code as open-source.*
        
        ---
        
        ## ⚖️ License / Lisensi
        
        Proyek ini menggunakan lisensi yang sama dengan repository aslinya. Silakan kunjungi [GitHub repository](https://github.com/obadx/quran-muaalem) untuk informasi lisensi lengkap.
        
        *This project uses the same license as the original repository. Please visit the [GitHub repository](https://github.com/obadx/quran-muaalem) for complete license information.*
        """)


def main(app=app):
    root_path = os.environ.get("GRADIO_ROOT_PATH", "/ngaji7")
    app.launch(server_name="0.0.0.0", share=False, root_path=root_path)
#    app.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()
    # app.launch(server_name="0.0.0.0", share=True)
