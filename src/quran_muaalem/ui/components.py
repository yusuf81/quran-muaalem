"""UI components and helper functions for Gradio interface."""

from typing import Any, Optional, get_origin
from pydantic.fields import FieldInfo, PydanticUndefined
import gradio as gr
from quran_transcript import Aya
from quran_transcript.utils import PartOfUthmaniWord
from quran_transcript.phonetics.moshaf_attributes import get_arabic_name

from ..config import sura_idx_to_name, sura_to_aya_count


def get_field_name(field_name: str, field_info: FieldInfo) -> str:
    """Return the Arabic name of the field if applicable else the field_name.
    
    Args:
        field_name: Field name in English
        field_info: Pydantic field info
        
    Returns:
        Formatted field label with Arabic translation if available
    """
    field_name_translations = {
        "rewaya": "Qiraah",
        "takbeer": "Takbir",
        "madd_monfasel_len": "Panjang Mad Munfasil",
        "madd_mottasel_len": "Panjang Mad Muttashil",
        "madd_mottasel_waqf": "Panjang Mad Muttashil pada Waqaf",
        "madd_aared_len": "Panjang Mad 'Aridh",
        "madd_alleen_len": "Panjang Mad Lin",
        "ghonna_lam_and_raa": "Ghunnah Lam dan Ra",
        "meem_aal_imran": "Mim Al Imran",
        "madd_yaa_alayn_alharfy": "Mad Ya Al-'Ain Al-Harfy",
        "saken_before_hamz": "Sakt sebelum Hamzah",
        "sakt_iwaja": "Sakt Iwaja",
        "sakt_marqdena": "Sakt Marqadena",
        "sakt_man_raq": "Sakt Man Raq",
        "sakt_bal_ran": "Sakt Bal Ran",
        "sakt_maleeyah": "Sakt Maleeyah",
        "between_anfal_and_tawba": "Antara Anfal dan At-Taubah",
        "noon_and_yaseen": "Nun dan Yasin",
        "yaa_ataan": "Ya' At-an",
        "start_with_ism": "Mulai dengan Basmalah",
        "yabsut": "Yabsut",
        "bastah": "Bastah",
        "almusaytirun": "Al-Musaytirun",
        "bimusaytir": "Bimusaytir",
        "tasheel_or_madd": "Tashil atau Mad",
        "yalhath_dhalik": "Yalhats Dhalik",
        "irkab_maana": "Irkab Ma'ana",
        "noon_tamnna": "Nun Tamnna",
        "harakat_daaf": "Harakat Dha'if",
        "alif_salasila": "Alif Salasila",
        "idgham_nakhluqkum": "Idgham Nakhluqkum",
        "raa_firq": "Ra' Firq",
        "raa_alqitr": "Ra' Al-Qitr",
        "raa_misr": "Ra' Mesir",
        "raa_nudhur": "Ra' Nudzur",
        "raa_yasr": "Ra' Yasr",
        "meem_mokhfah": "Mim Makhfi",
    }
    
    label = field_name_translations.get(field_name, field_name)
    arabic_name = get_arabic_name(field_info)
    if arabic_name:
        label = f"{arabic_name} ({label})"
    return label


def create_gradio_input_for_field(
    field_name: str,
    field_info: FieldInfo,
    default_value: Any = None,
    key_prefix="model_",
    help: str | None = None,
) -> Any:
    """Create appropriate Gradio input component for a Pydantic field.
    
    Args:
        field_name: Name of the field
        field_info: Pydantic field info
        default_value: Default value for the field
        key_prefix: Prefix for element ID
        help: Help text to display
        
    Returns:
        Gradio input component
    """
    label = get_field_name(field_name, field_info)
    
    # Use provided help text or field description
    if help is None:
        help = field_info.description or ""
    
    # Handle Literal (choices/dropdown)
    if get_origin(field_info.annotation) is type(None) or hasattr(field_info.annotation, '__args__'):
        # Check for Literal type
        try:
            from typing import Literal, get_args
            if get_origin(field_info.annotation) is Literal:
                choices = list(get_args(field_info.annotation))
                return gr.Dropdown(
                    choices=choices,
                    value=default_value or choices[0],
                    label=label,
                    info=help,
                    elem_id=f"{key_prefix}{field_name}"
                )
        except:
            pass
    
    # Handle basic types
    annotation = field_info.annotation
    if annotation in [str, Optional[str]]:
        return gr.Textbox(value=default_value or "", label=label, info=help)
    elif annotation in [int, Optional[int]]:
        return gr.Number(value=default_value or 0, label=label, info=help, precision=0)
    elif annotation in [float, Optional[float]]:
        return gr.Number(value=default_value or 0.0, label=label, info=help)
    elif annotation in [bool, Optional[bool]]:
        return gr.Checkbox(value=default_value or False, label=label, info=help)
    
    # Default to textbox
    return gr.Textbox(value=str(default_value) if default_value else "", label=label, info=help)


def update_aya_dropdown(sura_idx: int) -> gr.update:
    """Update ayah dropdown choices based on selected surah.
    
    Args:
        sura_idx: Surah index
        
    Returns:
        Gradio update object
    """
    if not sura_idx:
        sura_idx = 1
    return gr.update(
        choices=list(range(1, sura_to_aya_count[int(sura_idx)] + 1)), value=1
    )


def update_uthmani_ref(sura_idx: int, aya_idx: int) -> str:
    """Get Uthmani script for given surah and ayah.
    
    Args:
        sura_idx: Surah index
        aya_idx: Ayah index
        
    Returns:
        Uthmani text or error message
    """
    if not all([sura_idx, aya_idx]):
        return ""
    try:
        uthmani_ref = Aya(int(sura_idx), int(aya_idx)).get().uthmani
        return uthmani_ref
    except PartOfUthmaniWord as e:
        return (
            f"⚠️ Peringatan: Anda telah memilih sebagian kata Utsmani. "
            f"Silakan sesuaikan jumlah kata untuk hanya mencakup kata lengkap.\n\n"
            f"Detail kesalahan: {str(e)}"
        )
    except Exception as e:
        return f"Kesalahan: {str(e)}"


def update_uthmani_ref_html(sura_idx: int, aya_idx: int) -> str:
    """Get formatted HTML of Uthmani script.
    
    Args:
        sura_idx: Surah index
        aya_idx: Ayah index
        
    Returns:
        HTML formatted Uthmani text
    """
    text = update_uthmani_ref(sura_idx, aya_idx)
    if not text:
        return ""
    return (
        "<div style=\"font-size: 28px; line-height: 1.7; text-align: center; "
        'direction: rtl; font-family: "Scheherazade New", "Amiri", '
        '"Noto Naskh Arabic", "Traditional Arabic", serif;">'
        f"{text}"
        "</div>"
    )


def update_multi_verse_uthmani_preview(
    sura_idx: int,
    start_aya: int, 
    end_aya: int,
    full_sura_toggle: bool
) -> str:
    """Generate HTML preview for multiple verses.
    
    Args:
        sura_idx: Surah index
        start_aya: Start ayah number
        end_aya: End ayah number  
        full_sura_toggle: Whether analyzing full surah
        
    Returns:
        HTML formatted preview
    """
    try:
        if full_sura_toggle:
            start_aya = 1
            end_aya = sura_to_aya_count[int(sura_idx)]
        
        start_aya = int(start_aya)
        end_aya = int(end_aya)
        
        # Limit preview to first 20 verses
        MAX_PREVIEW = 20
        display_end = min(end_aya, start_aya + MAX_PREVIEW - 1)
        
        verses_html = ""
        for aya_num in range(start_aya, display_end + 1):
            try:
                uthmani_ref = Aya(int(sura_idx), aya_num).get().uthmani
                verses_html += f"""
                    <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
                        <div style='font-size: 14px; font-weight: bold; color: #666; margin-bottom: 5px;'>Ayat {aya_num}</div>
                        <div style='font-size: 24px; line-height: 1.7; text-align: center; direction: rtl; font-family: "Scheherazade New", "Amiri", "Noto Naskh Arabic", "Traditional Arabic", serif;'>{uthmani_ref}</div>
                    </div>
                    """
            except Exception as e:
                verses_html += f"<div style='color: red;'>Error loading Ayat {aya_num}: {str(e)}</div>"
        
        if end_aya > display_end:
            verses_html += f"<div style='color: #666; font-style: italic; margin-top: 10px;'>... dan {end_aya - display_end} ayat lainnya</div>"
        
        header = f"<h3>{sura_idx_to_name[int(sura_idx)]}: Ayat {start_aya}-{end_aya}</h3>"
        return header + verses_html
        
    except Exception as e:
        return f"<div style='color: red;'>Error: {str(e)}</div>"
