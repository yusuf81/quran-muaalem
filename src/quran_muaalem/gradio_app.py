import logging
from dataclasses import asdict
import json
from typing import Literal, Optional, Any, get_origin, get_args

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from quran_transcript.utils import PartOfUthmaniWord
from quran_transcript.phonetics.moshaf_attributes import (
    get_arabic_attributes,
    get_arabic_name,
)
from librosa.core import load
from pydantic.fields import FieldInfo, PydanticUndefined
import torch
import gradio as gr

from quran_muaalem.inference import Muaalem
from quran_muaalem.muaalem_typing import MuaalemOutput
from quran_muaalem.explain import explain_for_terminal
from quran_muaalem.explain_gradio import explain_for_gradio

# Initialize components
REQUIRED_MOSHAF_FIELDS = [
    "rewaya",
    "takbeer",
    "madd_monfasel_len",
    "madd_mottasel_len",
    "madd_mottasel_waqf",
    "madd_aared_len",
    "madd_alleen_len",
    "ghonna_lam_and_raa",
    "meem_aal_imran",
    "madd_yaa_alayn_alharfy",
    "saken_before_hamz",
    "sakt_iwaja",
    "sakt_marqdena",
    "sakt_man_raq",
    "sakt_bal_ran",
    "sakt_maleeyah",
    "between_anfal_and_tawba",
    "noon_and_yaseen",
    "yaa_ataan",
    "start_with_ism",
    "yabsut",
    "bastah",
    "almusaytirun",
    "bimusaytir",
    "tasheel_or_madd",
    "yalhath_dhalik",
    "irkab_maana",
    "noon_tamnna",
    "harakat_daaf",
    "alif_salasila",
    "idgham_nakhluqkum",
    "raa_firq",
    "raa_alqitr",
    "raa_misr",
    "raa_nudhur",
    "raa_yasr",
    "meem_mokhfah",
]
model_id = "obadx/muaalem-model-v3_2"
logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
muaalem = Muaalem(model_name_or_path=model_id, device=device)
sampling_rate = 16000

# Load Sura information
sura_idx_to_name = {}
sura_to_aya_count = {}
start_aya = Aya()
for sura_idx in range(1, 115):
    start_aya.set(sura_idx, 1)
    sura_idx_to_name[sura_idx] = start_aya.get().sura_name
    sura_to_aya_count[sura_idx] = start_aya.get().num_ayat_in_sura

# Default moshaf settings
default_moshaf = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=4,
)

# Current moshaf settings (will be updated from settings page)
current_moshaf = default_moshaf


def get_field_name(field_name: str, field_info: FieldInfo) -> str:
    """Return the Arabic name of the field if applicable else the field_name"""
    # Terjemahan nama field ke bahasa Indonesia
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
    """Create a gradio input field given a pydantic field info"""
    # Extract Arabic name from field description if available
    label = get_field_name(field_name, field_info)

    if default_value is None:
        if field_info.default != PydanticUndefined:
            default_value = field_info.default

    if help is None:
        help = field_info.description

    # Terjemahkan deskripsi jika tersedia
    if help:
        # Kamus terjemahan untuk deskripsi
        help_translations = {
            "The Rewaya to use for recitation.": "Qiraah yang digunakan untuk bacaan.",
            "The length of Mad Al Monfasel \"مد النفصل\" for Hafs Rewaya.": "Panjang Mad Al Munfasil \"مد النفصل\" untuk Qiraah Hafs.",
            "The length of Mad Al Mottasel \"مد المتصل\" for Hafs Rewaya.": "Panjang Mad Al Muttashil \"مد المتصل\" untuk Qiraah Hafs.",
            "The length of Mad Al Mottasel \"مد المتصل\" on Waqf for Hafs Rewaya.": "Panjang Mad Al Muttashil \"مد المتصل\" pada akhir ayat untuk Qiraat Hafs.",
            "The length of Mad Al Aared \"مد العارض\" for Hafs Rewaya.": "Panjang Mad Al 'Aridh \"مد العارض\" untuk Qiraat Hafs.",
            "The length of Mad Alleen \"مد اللين\" for Hafs Rewaya.": "Panjang Mad Al-Leen \"مد اللين\" untuk Qiraat Hafs.",
            "The ghonna for merging (Idghaam) noon with Lam and Raa for Hafs.": "Ghunnah untuk idgham (peluluran) nun dengan Lam dan Ra untuk Hafs.",
            "Enable Ghonna on Lam and Raa when Sakien with 2 counts.": "Aktifkan Ghunnah pada Lam dan Ra ketika sukun selama 2 ketukan.",
            "Enable Meem Al Imran with 2 counts.": "Aktifkan Mim Al Imran selama 2 ketukan.",
            "Enable Mad Yaa Alayn Alharfy with 2 counts.": "Aktifkan Mad Ya Al-'Ain Al-Harfy selama 2 hitungan.",
            "Enable Saken before Hamz with 1 or 2 counts.": "Aktifkan sakt sebelum hamzah selama 1 atau 2 hitungan.",
            "Enable Sakt Iwaja with 1 or 2 counts.": "Aktifkan sakt Iwaja selama 1 atau 2 hitungan.",
            "Enable Sakt Marqdena with 1 or 2 counts.": "Aktifkan sakt Marqadena selama 1 atau 2 hitungan.",
            "Enable Sakt Man Raq with 1 or 2 counts.": "Aktifkan sakt Man Raq selama 1 atau 2 hitungan.",
            "Enable Sakt Bal Ran with 1 or 2 counts.": "Aktifkan sakt Bal Ran selama 1 atau 2 hitungan.",
            "Enable Sakt Maleeyah with 1 or 2 counts.": "Aktifkan sakt Maleeyah selama 1 atau 2 hitungan.",
            "Enable between Anfal and Tawba without Basmala.": "Aktifkan antara Anfal dan At-Taubah tanpa basmalah.",
            "Enable Noon and Yaseen.": "Aktifkan Nun dan Yasin.",
            "Enable Yaa At-an.": "Aktifkan Ya' At-an.",
            "Enable start with ism.": "Aktifkan awali dengan basmalah.",
            "Enable Yabsut.": "Aktifkan Yabsut.",
            "Enable Bastah.": "Aktifkan Bastah.",
            "Enable Almusaytirun.": "Aktifkan Al-Musaytirun.",
            "Enable Bimusaytir.": "Aktifkan Bimusaytir.",
            "Enable Tasheel or Madd.": "Aktifkan tashil atau mad.",
            "Enable Yalhath Dhalik.": "Aktifkan Yalhats Dhalik.",
            "Enable Irkab Maana.": "Aktifkan Irkab Ma'ana.",
            "Enable Noon Tamnna.": "Aktifkan Nun Tamnna.",
            "Enable Harakat Daaf.": "Aktifkan harakat dha'if.",
            "Enable Alif Salasila.": "Aktifkan Alif Salasila.",
            "Enable Idgham Nakhluqkum.": "Aktifkan idgham Nakhluqkum.",
            "Enable Raa Firq.": "Aktifkan Ra' Firq.",
            "Enable Raa Alqitr.": "Aktifkan Ra' Al-Qitr.",
            "Enable Raa Misr.": "Aktifkan Ra' Mesir.",
            "Enable Raa Nudhur.": "Aktifkan Ra' Nudzur.",
            "Enable Raa Yasr.": "Aktifkan Ra' Yasr.",
            "Enable Meem Mokhfah.": "Aktifkan Mim Makhfi.",
            # Tambahkan terjemahan untuk deskripsi takbir
            'The ways to add takbeer (الله أكبر) after Istiaatha (استعاذة) and between end of the surah and beginning of the surah. no_takbeer: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation) beginning_of_sharh: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas end_of_dohaf: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas general_takbeer: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah': 'Cara menambahkan takbir (الله أكبر) setelah Istiaatha (استعاذة) dan antara akhir surah dan awal surah. no_takbeer: "لا تكبير" — Tanpa Takbir (Tanpa pengucapan kebesaran, yaitu tidak ada bacaan Takbir) beginning_of_sharh: "التكبير من أول الشرح لأول الناس" — Takbir dari awal Surah Ash-Sharh ke awal Surah An-Nas end_of_dohaf: "التكبير من آخر الضحى لآخر الناس" — Takbir dari akhir Surah Ad-Duha ke akhir Surah An-Nas general_takbeer: "التكبير أول كل سورة إلا التوبة" — Takbir di awal setiap Surah kecuali Surah At-Tawbah',
            # Tambahkan entri terjemahan untuk deskripsi lainnya dari MoshafAttributes
            ' The length of Mad Al Monfasel "مد النفصل" for Hafs Rewaya.': 'Panjang Mad Al Monfasel (مد النفصل) untuk qiraah Hafs.',
            ' The length of Mad Al Motasel "مد المتصل" for Hafs.': 'Panjang Mad Al Muttasil (مد المتصل) untuk qiraah Hafs.',
            ' The length of Madd Almotasel at pause for Hafs. Example "السماء".': 'Panjang Mad Muttasil pada akhir ayat untuk qira\'ah Hafs. Contoh "السماء".',
            ' The length of Mad Al Aared "مد العارض للسكون".': 'Panjang Mad Al Aared (مد العارض للسكون) untuk qiraah Hafs.',
            'The length of the Madd al-Leen when stopping at the end of a word (for a sakin waw or ya preceded by a letter with a fatha) should be less than or equal to the length of Madd al-\'Arid (the temporary stretch due to stopping). **Default Value is equal to `madd_aared_len`**. مقدرا مع اللين عن القوف (للواو الساكنة والياء الساكنة وقبلها حرف مفتوح) ويجب أن يكون مقدار مد اللين أقل من أو يساوي مع العارض': 'Panjang Mad Al-Leen saat berhenti di akhir kata (untuk wawu sukun atau ya\' yang didahului oleh huruf dengan fathah) seharusnya kurang dari atau sama dengan panjang Mad Al-Arid (pemanjangan sementara karena berhenti). **Nilai bawaan sama dengan `madd_aared_len`**. Nilai Mad Lin saat berhenti (untuk wawu sukun atau ya sukun dengan huruf fathah sebelumnya) dan nilai Mad Lin harus kurang dari atau sama dengan Mad Aared.',
            'The ghonna for merging (Idghaam) noon with Lam and Raa for Hafs.': 'Ghunnah untuk menggabung (Idgham) nun dengan Lam dan Ra untuk Hafs.',
            'The ways to recite the word meem Aal Imran (الم الله) at connected recitation. `waqf`: Pause with a prolonged madd (elongation) of 6 harakat (beats). `wasl_2` Pronounce "meem" with fathah (a short "a" sound) and stretch it for 2 harakat. `wasl_6` Pronounce "meem" with fathah and stretch it for 6 harakat.': 'Cara membaca kata meem Aal Imran (الم الله) pada bacaan bersambung. `waqf`: Berhenti dengan mad panjang (pemanjangan) selama 6 harakat (ketukan). `wasl_2` Ucapkan "meem" dengan fathah (suara pendek "a") dan ulurkan selama 2 harakat. `wasl_6` Ucapkan "meem" dengan fathah dan ulurkan selama 6 harakat.',
            ' The length of Lzem Harfy of Yaa in letter Al-Ayen Madd "المد الحرفي اللازم لحرف العين" in surar: Maryam "مريم", AlShura "الشورى".': 'Panjang Lzem Harfy dari Ya dalam huruf Al-Ayen Madd "المد الحرفي اللازم لحرف العين" dalam surah: Maryam "مريم", AlShura "الشورى".',
            'The ways of Hafs for saken before hamz. "The letter with sukoon before the hamzah (ء)".And it has three forms: full articulation (`tahqeeq`), general pause (`general_sakt`), and specific pause (`local_skat`).': 'Cara Hafs untuk sakt sebelum hamz. "Huruf dengan sukun sebelum hamzah (ء)".Dan memiliki tiga bentuk: penuh (`tahqeeq`), jeda umum (`general_sakt`), dan jeda khusus (`local_skat`).',
            'The ways to recite the word "عوجا" (Iwaja). `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).': 'Cara membaca kata "عوجا" (Iwaja). `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).',
            'The ways to recite the word "مرقدنا" (Marqadena) in Surat Yassen. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).': 'Cara membaca kata "مرقدنا" (Marqadena) dalam Surat Yasin. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).',
            'The ways to recite the word "من راق" (Man Raq) in Surat Al Qiyama. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).': 'Cara membaca kata "من راق" (Man Raq) dalam Surat Al Qiyama. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).',
            'The ways to recite the word "بل ران" (Bal Ran) in Surat Al Motaffin. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).': 'Cara membaca kata "بل ران" (Bal Ran) dalam Surat Al Motaffin. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).',
            'The ways to recite the word {ماليه هلك} in Surah Al-Ahqaf. `sakt` means slight pause. `idgham` Assimilation of the letter \'Ha\' (ه) into the letter \'Ha\' (ه) with complete assimilation.`waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idgham`.': 'Cara membaca kata {ماليه هلك} dalam Surah Al-Ahqaf. `sakt` artinya jeda sedikit. `idgham` Pelumeran huruf \'Ha\' (ه) ke huruf \'Ha\' (ه) dengan pelumeran lengkap.`waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idgham`.',
            'The ways to recite end of Surah Al-Anfal and beginning of Surah At-Tawbah.': 'Cara membaca akhir Surah Al-Anfal dan awal Surah At-Tawbah.',
            'Weither to merge noon of both: {يس} and {ن} with (و) "`idgham`" or not "`izhar`".': 'Apakah menggabungkan nun dari kedua: {يس} dan {ن} dengan (و) "`idgham`" atau tidak "`izhar`".',
            'The affirmation and omission of the letter \'Yaa\' in the pause of the verse {آتاني} in Surah An-Naml.`wasl`: means connected recitation without pasuding as (آتانيَ).`hadhf`: means deletion of letter (ي) at puase so recited as (آتان).`ithbat`: means confirmation reciting letter (ي) at puase as (آتاني).': 'Pengakuan dan pengabaian huruf \'Yaa\' dalam jeda ayat {آتани} dalam Surah An-Naml.`wasl`: artinya bacaan bersambung tanpa jeda seperti (آتانيَ).`hadhf`: artinya penghapusan huruf (ي) saat jeda jadi dibaca (آتان).`ithbat`: artinya konfirmasi membaca huruf (ي) saat jeda seperti (آتاني).',
            'The ruling on starting with the word {الاسم} in Surah Al-Hujurat.`lism` Recited as (لسم) at the beginning. `alism` Recited as (ألسم). ath the beginning`wasl`: means completing recitaion without paussing as normal, So Reciting is as (بئس لسم).': 'Hukum memulai dengan kata {الاسم} dalam Surah Al-Hujurat.`lism` Dibaca sebagai (لسم) di awal. `alism` Dibaca sebagai (ألسم). di awal`wasl`: artinya menyelesaikan bacaan tanpa jeda seperti biasa, Jadi dibaca sebagai (بئس لسم).',
            'The ruling on pronouncing `seen` (س) or `saad` (ص) in the verse {والله يقبض ويبسط} in Surah Al-Baqarah.': 'Hukum mengucapkan `sin` (س) atau `shad` (ص) dalam ayat {والله يقبض ويبسط} dalam Surah Al-Baqarah.',
            'The ruling on pronouncing `seen` (س) or `saad` (ص ) in the verse {وزادكم في الخلق بسطة} in Surah Al-A\'raf.': 'Hukum mengucapkan `sin` (س) atau `shad` (ص) dalam ayat {وزادكم في الخلق بسطة} dalam Surah Al-A\'raf.',
            'The pronunciation of `seen` (س) or `saad` (ص ) in the verse {أم هم المصيطرون} in Surah At-Tur.': 'Pengucapan `sin` (س) atau `shad` (ص) dalam ayat {أم هم المصيطرون} dalam Surah At-Tur.',
            'The pronunciation of `seen` (س) or `saad` (ص ) in the verse {لست عليهم بمصيطر} in Surah Al-Ghashiyah.': 'Pengucapan `sin` (س) atau `shad` (ص) dalam ayat {لست عليهم بمصيطر} dalam Surah Al-Ghashiyah.',
            ' Tasheel of Madd "وجع التسهيل أو المد" for 6 words in The Holy Quran: "ءالذكرين", "ءالله", "ءائن".': 'Tashil dari Mad "وجع التسهيل أو المد" untuk 6 kata dalam Al-Quran Mulia: "ءالذكرين", "ءالله", "ءائن".',
            'The assimilation (`idgham`) and non-assimilation (`izhar`) in the verse {يلهث ذلك} in Surah Al-A\'raf. `waqf`: means the rectier has paused on (يلهث)': 'Pelumeran (`idgham`) dan bukan pelumeran (`izhar`) dalam ayat {يلهث ذلك} dalam Surah Al-A\'raf. `waqf`: artinya pembaca telah berhenti pada (يلهث)',
            'The assimilation and clear pronunciation in the verse {اركب معنا} in Surah Hud.This refers to the recitation rules concerning whether the letter "Noon" (ن) is assimilated into the following letter or pronounced clearly when reciting this specific verse. `waqf`: means the rectier has paused on (اركب)': 'Pelumeran dan pengucapan jelas dalam ayat {اركب معنا} dalam Surah Hud. Ini mengacu pada aturan bacaan tentang apakah huruf "Nun" (ن) dilumerkan ke huruf berikutnya atau diucapkan dengan jelas saat membaca ayat ini secara khusus. `waqf`: artinya pembaca telah berhenti pada (اركب)',
            'The nasalization (`ishmam`) or the slight drawing (`rawm`) in the verse {لا تأمنا على يوسف}': 'Ghunnah (`ishmam`) atau penarikan sedikit (`rawm`) dalam ayat {لا تأمنا على يوسف}',
            'The vowel movement of the letter \'Dhad\' (ض) (whether with `fath` or `dam`) in the word {ضعف} in Surah Ar-Rum.': 'Gerakan vokal huruf \'Dhad\' (ض) (apakah dengan `fathah` atau `dhammah`) dalam kata {ضعف} dalam Surah Ar-Rum.',
            'Affirmation and omission of the \'Alif\' when pausing in the verse {سلاسلا} in Surah Al-Insan.This refers to the recitation rule regarding whether the final "Alif" in the word "سلاسلا" is pronounced (affirmed) or omitted when pausing (waqf) at this word during recitation in the specific verse from Surah Al-Insan. `hadhf`: means to remove alif (ا) during puase as (سلاسل) `ithbat`: means to recite alif (ا) during puase as (سلاسلا) `wasl` means completing the recitation as normal without pausing, so recite it as (سلاسلَ وأغلالا)': 'Pengakuan dan pengabaian \'Alif\' saat berhenti dalam ayat {سلاسلا} dalam Surah Al-Insan. Ini mengacu pada aturan bacaan tentang apakah "Alif" terakhir dalam kata "سلاسلا" diucapkan (dikonfirmasi) atau diabaikan saat berhenti (waqf) di kata ini selama bacaan dalam ayat spesifik dari Surah Al-Insan. `hadhf`: artinya menghapus alif (ا) saat berhenti seperti (سلاسل) `ithbat`: artinya membaca alif (ا) saat berhenti seperti (سلاسلا) `wasl` artinya menyelesaikan bacaan secara normal tanpa berhenti, jadi bacalah sebagai (سلاسلَ وأغلالا)',
            'Assimilation of the letter \'Qaf\' into the letter \'Kaf,\' whether incomplete (`idgham_naqis`) or complete (`idgham_kamil`), in the verse {نخلقكم} in Surah Al-Mursalat.': 'Pelumeran huruf \'Qaf\' ke huruf \'Kaf\', apakah tidak lengkap (`idgham_naqis`) atau lengkap (`idgham_kamil`), dalam ayat {نخلقكم} dalam Surah Al-Mursalat.',
            'Emphasis and softening of the letter \'Ra\' in the word {urfq} in Surah Ash-Shu\'ara\' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "urfq"  is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the specific verse from Surah Ash-Shu\'ara\' in connected speech. `waqf`: means pasuing so we only have one way (tafkheem of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {urfq} dalam Surah Ash-Shu\'ara\' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "urfq" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat membaca ayat spesifik dari Surah Ash-Shu\'ara\' secara bacaan bersambung. `waqf`: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)',
            'Emphasis and softening of the letter \'Ra\' in the word {فرق} in Surah Ash-Shu\'ara\' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "فرق"  is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the specific verse from Surah Ash-Shu\'ara\' in connected speech. `waqf`: means pasuing so we only have one way (tafkheem of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {فرق} dalam Surah Ash-Shu\'ara\' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "فرق" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat membaca ayat spesifik dari Surah Ash-Shu\'ara\' secara bacaan bersambung. `waqf`: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)',
            'Emphasis and softening of the letter \'Ra\' in the word {فرق} in Surah Ash-Shu\'ara\' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "فرق" is pronounced with emphasis (tafkheem) or softening (tarqeeq) when reciting the specific verse from Surah Ash-Shu\'ara\' in connected speech. waqf: means pasuing so we only have one way (tafkheem of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {فرق} dalam Surah Ash-Shu\'ara\' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "فرق" diucapkan dengan penekanan (tafkheem) atau pelunakan (tarqeeq) saat membaca ayat spesifik dari Surah Ash-Shu\'ara\' secara bacaan bersambung. waqf: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)',
            'Emphasis and softening of the letter \'Ra\' in the word {فرق} in Surah Ash-Shu\'ara\' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "فرق" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the specific verse from Surah Ash-Shu\'ara\' in connected speech. `waqf`: means pasuing so we only have one way (tafkheem of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {فرق} dalam Surah Ash-Shu\'ara\' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "فرق" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat membaca ayat spesifik dari Surah Ash-Shu\'ara\' secara bacaan bersambung. `waqf`: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)',
            'Emphasis and softening of the letter \'Ra\' in the word {القطر} in Surah Saba\' when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "القطر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Saba\'. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {القطر} dalam Surah Saba\' saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "القطر" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Saba\'. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)',
            'Emphasis and softening of the letter \'Ra\' in the word {مصر} in Surah Yunus, and in the locations of Surah Yusuf and Surah Az-Zukhruf when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "مصر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) at the specific pauses in these Surahs. `wasl`: means not pasuing so we only have one way (tafkheem of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {مصر} dalam Surah Yunus, dan dalam lokasi Surah Yusuf dan Surah Az-Zukhruf saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "مصر" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) pada posisi berhenti spesifik dalam Surah-surah ini. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)',
            'Emphasis and softening of the letter \'Ra\' in the word {نذر} in Surah Al-Qamar when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "نذر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Al-Qamar. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {نذر} dalam Surah Al-Qamar saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "نذر" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Al-Qamar. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)',
            'Emphasis and softening of the letter \'Ra\' in the word {يسر} in Surah Al-Fajr when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "يسر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Al-Fajr. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {يسر} dalam Surah Al-Fajr saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "يسر" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Al-Fajr. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)',
            'This is not a standrad Hafs way but a disagreement between schoolars in our century how to pronounc Ikhfaa for meem. Some schoolars do full merging `إدام` and the other open the leaps a little bit `إخفاء`. We did not want to add this but some of the best reciters disagree about this': 'Ini bukan cara standar Hafs tetapi perbedaan pendapat antara ulama di abad kita bagaimana mengucapkan Ikhfaa untuk meem. Beberapa ulama melakukan pelumeran penuh `إدام` dan yang lain membuka celah sedikit `إخفاء`. Kita tidak ingin menambahkan ini tetapi beberapa pembaca terbaik memiliki perbedaan pendapat tentang hal ini',
            # Tambahkan deskripsi dasar
            "The type of the quran Rewaya.": "Jenis qiraah Al-Quran.",
            # Tambahkan entri untuk deskripsi yang belum ditangani
            'Emphasis and softening of the letter \'Ra\' in the word {urfq} in Surah Ash-Shu\'ara\' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "urfq"  is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the specific verse from Surah Ash-Shu\'ara\' in connected speech. `waqf`: means pasuing so we only have one way (tafkheem of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {urfq} dalam Surah Ash-Shu\'ara\' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "urfq" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat membaca ayat spesifik dari Surah Ash-Shu\'ara\' secara bacaan bersambung. `waqf`: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)',
            # Tambahkan entri untuk deskripsi yang sesuai dengan teks HTML yang ditampilkan
            'The ways to add takbeer (الله أكبر) after Istiaatha (استعاذة) and between end of the surah and beginning of the surah. <code>no_takbeer</code>: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah': 'Cara menambahkan takbir (الله أكبر) setelah Istiaatha (استعاذة) dan antara akhir surah dan awal surah. <code>no_takbeer</code>: "لا تكبير" — Tanpa Takbir (Tanpa pengucapan kebesaran, yaitu tidak ada bacaan Takbir) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbir dari awal Surah Ash-Sharh ke awal Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbir dari akhir Surah Ad-Duha ke akhir Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbir di awal setiap Surah kecuali Surah At-Tawbah',
            "The length of Madd Almotasel at pause for Hafs.. Example \"السماء\".": "Panjang Mad Muttasil pada akhir ayat untuk qira'ah Hafs. Contoh \"السماء\".",
            # Tambahkan entri untuk kalimat yang sama seperti muncul dalam HTML
            'The ways to add takbeer (الله أكبر) after Istiaatha (استعاذة) and between end of the surah and beginning of the surah. <code>no_takbeer</code>: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah': 'Cara menambahkan takbir (الله أكبر) setelah Istiaatha (استعاذة) dan antara akhir surah dan awal surah. <code>no_takbeer</code>: "لا تكبير" — Tanpa Takbir (Tanpa pengucapan kebesaran, yaitu tidak ada bacaan Takbir) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbir dari awal Surah Ash-Sharh ke awal Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbir dari akhir Surah Ad-Duha ke akhir Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbir di awal setiap Surah kecuali Surah At-Tawbah',
            # Tambahkan juga entri dengan format yang persis seperti muncul dalam HTML Anda
            'The ways to add takbeer (الله أكبر) after Istiaatha (استعاذة) and between end of the surah and beginning of the surah. <code>no_takbeer</code>: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah': 'Cara menambahkan takbir (الله أكبر) setelah Istiaatha (استعاذة) dan antara akhir surah dan awal surah. <code>no_takbeer</code>: "لا تكبير" — Tanpa Takbir (Tanpa pengucapan kebesaran, yaitu tidak ada bacaan Takbir) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbir dari awal Surah Ash-Sharh ke awal Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbir dari akhir Surah Ad-Duha ke akhir Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbir di awal setiap Surah kecuali Surah At-Tawbah',
            # Tambahkan entri untuk teks deskripsi spesifik dari HTML
            'The length of Madd Almotasel at pause for Hafs.. Example "السماء".': 'Panjang Mad Muttasil pada akhir ayat untuk qira\'ah Hafs. Contoh "السماء".',
            # Tambahkan entri untuk teks yang persis muncul di HTML
            'The length of Mad Al Motasel "مد المتصل" for Hafs.': 'Panjang Mad Al Muttasil (مد المتصل) untuk qira\'ah Hafs.',
            'The length of Madd Almotasel at pause for Hafs.. Example "السماء".': 'Panjang Mad Muttasil pada akhir ayat untuk qira\'ah Hafs. Contoh "السماء".',
            'The length of Mad Almotasel at pause for Hafs.. Example "السماء".': 'Panjang Mad Muttasil pada akhir ayat untuk qira\'ah Hafs. Contoh "السماء".',
            # Tambahkan entri untuk deskripsi huruf Ra yang bermasalah
            'Emphasis and softening of the letter \'Ra\' in the word {urfq} in Surah Ash-Shu\'ara\' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "urfq"  is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the specific verse from Surah Ash-Shu\'ara\' in connected speech. `waqf`: means pasuing so we only have one way (tafkheem of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {urfq} dalam Surah Ash-Shu\'ara\' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "urfq" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat membaca ayat spesifik dari Surah Ash-Shu\'ara\' secara bacaan bersambung. `waqf`: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)',
            # Tambahkan entri berdasarkan deskripsi aktual dari file sumber
            'The ways to add takbeer (الله أكبر) after Istiaatha (استعاذة) and between end of the surah and beginning of the surah. `no_takbeer`: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation) `beginning_of_sharh`: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas `end_of_dohaf`: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas `general_takbeer`: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah': 'Cara menambahkan takbir (الله أكبر) setelah Istiaatha (استعاذة) dan antara akhir surah dan awal surah. `no_takbeer`: "لا تكبير" — Tanpa Takbir (Tanpa pengucapan kebesaran, yaitu tidak ada bacaan Takbir) `beginning_of_sharh`: "التكبير من أول الشرح لأول الناس" — Takbir dari awal Surah Ash-Sharh ke awal Surah An-Nas `end_of_dohaf`: "التكبير من آخر الضحى لآخر الناس" — Takbir dari akhir Surah Ad-Duha ke akhir Surah An-Nas `general_takbeer`: "التكبير أول كل سورة إلا التوبة" — Takbir di awal setiap Surah kecuali Surah At-Tawbah',
            ' The length of Mad Al Monfasel "مد النفصل" for Hafs Rewaya.': 'Panjang Mad Al Monfasel (مد النفصل) untuk qira\'ah Hafs.',
            ' The length of Mad Al Motasel "مد المتصل" for Hafs.': 'Panjang Mad Al Muttasil (مد المتصل) untuk qira\'ah Hafs.',
            " The length of Madd Almotasel at pause for Hafs.. Example \"السماء\".": "Panjang Mad Muttasil pada akhir ayat untuk qiraah Hafs. Contoh \"السماء\".",
            ' The length of Mad Al Aared "مد العارض للسكون".': 'Panjang Mad Al Aared (مد العارض للسكون) untuk qira\'ah Hafs.',
            "The length of the Madd al-Leen when stopping at the end of a word (for a sakin waw or ya preceded by a letter with a fatha) should be less than or equal to the length of Madd al-'Arid (the temporary stretch due to stopping). **Default Value is equal to `madd_aared_len`**. مقدرا مع اللين عن القوف (للواو الساكنة والياء الساكنة وقبلها حرف مفتوح) ويجب أن يكون مقدار مد اللين أقل من أو يساوي مع العارض": "Panjang Mad Al-Leen saat berhenti di akhir kata (untuk wawu sukun atau ya' yang didahului oleh huruf dengan fathah) seharusnya kurang dari atau sama dengan panjang Mad Al-Arid (pemanjangan sementara karena berhenti). **Nilai bawaan sama dengan `madd_aared_len`**. Nilai Mad Lin saat berhenti (untuk wawu sukun atau ya sukun dengan huruf fathah sebelumnya) dan nilai Mad Lin harus kurang dari atau sama dengan Mad Aared.",
            "The ghonna for merging (Idghaam) noon with Lam and Raa for Hafs.": "Ghunnah untuk menggabung (Idgham) nun dengan Lam dan Ra untuk Hafs.",
            "The ways to recite the word meem Aal Imran (الم الله) at connected recitation. `waqf`: Pause with a prolonged madd (elongation) of 6 harakat (beats). `wasl_2` Pronounce \"meem\" with fathah (a short \"a\" sound) and stretch it for 2 harakat. `wasl_6` Pronounce \"meem\" with fathah and stretch it for 6 harakat.": "Cara membaca kata meem Aal Imran (الم الله) pada bacaan bersambung. `waqf`: Berhenti dengan mad panjang (pemanjangan) selama 6 harakat (ketukan). `wasl_2` Ucapkan \"meem\" dengan fathah (suara pendek \"a\") dan ulurkan selama 2 harakat. `wasl_6` Ucapkan \"meem\" dengan fathah dan ulurkan selama 6 harakat.",
            " The length of Lzem Harfy of Yaa in letter Al-Ayen Madd \"المد الحرفي اللازم لحرف العين\" in surar: Maryam \"مريم\", AlShura \"الشورى\".": "Panjang Lzem Harfy dari Ya dalam huruf Al-Ayen Madd \"المد الحرفي اللازم لحرف العين\" dalam surah: Maryam \"مريم\", AlShura \"الشورى\".",
            "The ways of Hafs for saken before hamz. \"The letter with sukoon before the hamzah (ء)\".And it has three forms: full articulation (`tahqeeq`), general pause (`general_sakt`), and specific pause (`local_skat`).": "Cara Hafs untuk sakt sebelum hamz. \"Huruf dengan sukun sebelum hamzah (ء)\".Dan memiliki tiga bentuk: penuh (`tahqeeq`), jeda umum (`general_sakt`), dan jeda khusus (`local_skat`).",
            "The ways to recite the word \"عوجا\" (Iwaja). `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).": "Cara membaca kata \"عوجا\" (Iwaja). `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).",
            "The ways to recite the word \"مرقدنا\" (Marqadena) in Surat Yassen. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).": "Cara membaca kata \"مرقدنا\" (Marqadena) dalam Surat Yasin. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).",
            "The ways to recite the word \"من راق\" (Man Raq) in Surat Al Qiyama. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).": "Cara membaca kata \"من راق\" (Man Raq) dalam Surat Al Qiyama. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).",
            "The ways to recite the word \"بل ران\" (Bal Ran) in Surat Al Motaffin. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).": "Cara membaca kata \"بل ران\" (Bal Ran) dalam Surat Al Motaffin. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).",
            "The ways to recite the word {ماليه هلك} in Surah Al-Ahqaf. `sakt` means slight pause. `idgham` Assimilation of the letter 'Ha' (ه) into the letter 'Ha' (ه) with complete assimilation.`waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idgham`.": "Cara membaca kata {ماليه هلك} dalam Surah Al-Ahqaf. `sakt` artinya jeda sedikit. `idgham` Pelumeran huruf 'Ha' (ه) ke huruf 'Ha' (ه) dengan pelumeran lengkap.`waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idgham`.",
            "The ways to recite end of Surah Al-Anfal and beginning of Surah At-Tawbah.": "Cara membaca akhir Surah Al-Anfal dan awal Surah At-Tawbah.",
            'Weither to merge noon of both: {يس} and {ن} with (و) "`idgham`" or not "`izhar`".': 'Apakah menggabungkan nun dari kedua: {يس} dan {ن} dengan (و) "`idgham`" atau tidak "`izhar`".',
            "The affirmation and omission of the letter 'Yaa' in the pause of the verse {آتاني} in Surah An-Naml.`wasl`: means connected recitation without pasuding as (آتانيَ).`hadhf`: means deletion of letter (ي) at puase so recited as (آتان).`ithbat`: means confirmation reciting letter (ي) at puase as (آتاني).": "Pengakuan dan pengabaian huruf 'Yaa' dalam jeda ayat {آتاني} dalam Surah An-Naml.`wasl`: artinya bacaan bersambung tanpa jeda seperti (آتانيَ).`hadhf`: artinya penghapusan huruf (ي) saat jeda jadi dibaca (آتان).`ithbat`: artinya konfirmasi membaca huruf (ي) saat jeda seperti (آتاني).",
            "The ruling on starting with the word {الاسم} in Surah Al-Hujurat.`lism` Recited as (لسم) at the beginning. `alism` Recited as (ألسم). ath the beginning`wasl`: means completing recitaion without paussing as normal, So Reciting is as (بئس لسم).": "Hukum memulai dengan kata {الاسم} dalam Surah Al-Hujurat.`lism` Dibaca sebagai (لسم) di awal. `alism` Dibaca sebagai (ألسم). di awal`wasl`: artinya menyelesaikan bacaan tanpa jeda seperti biasa, Jadi dibaca sebagai (بئس لسم).",
            "The ruling on pronouncing `seen` (س) or `saad` (ص) in the verse {والله يقبض ويبسط} in Surah Al-Baqarah.": "Hukum mengucapkan `sin` (س) atau `shad` (ص) dalam ayat {والله يقبض ويبسط} dalam Surah Al-Baqarah.",
            "The ruling on pronouncing `seen` (س) or `saad` (ص ) in the verse {وزادكم في الخلق بسطة} in Surah Al-A'raf.": "Hukum mengucapkan `sin` (س) atau `shad` (ص) dalam ayat {وزادكم في الخلق بسطة} dalam Surah Al-A'raf.",
            "The pronunciation of `seen` (س) or `saad` (ص ) in the verse {أم هم المصيطرون} in Surah At-Tur.": "Pengucapan `sin` (س) atau `shad` (ص) dalam ayat {أم هم المصيطرون} dalam Surah At-Tur.",
            "The pronunciation of `seen` (س) or `saad` (ص ) in the verse {لست عليهم بمصيطر} in Surah Al-Ghashiyah.": "Pengucapan `sin` (س) atau `shad` (ص) dalam ayat {لست عليهم بمصيطر} dalam Surah Al-Ghashiyah.",
            " Tasheel of Madd \"وجع التسهيل أو المد\" for 6 words in The Holy Quran: \"ءالذكرين\", \"ءالله\", \"ءائن\".": "Tashil dari Mad \"وجع التسهيل أو المد\" untuk 6 kata dalam Al-Quran Mulia: \"ءالذكرين\", \"ءالله\", \"ءائن\".",
            "The assimilation (`idgham`) and non-assimilation (`izhar`) in the verse {يلهث ذلك} in Surah Al-A'raf. `waqf`: means the rectier has paused on (يلهث)": "Pelumeran (`idgham`) dan bukan pelumeran (`izhar`) dalam ayat {يلهث ذلك} dalam Surah Al-A'raf. `waqf`: artinya pembaca telah berhenti pada (يلهث)",
            "The assimilation and clear pronunciation in the verse {اركب معنا} in Surah Hud.This refers to the recitation rules concerning whether the letter \"Noon\" (ن) is assimilated into the following letter or pronounced clearly when reciting this specific verse. `waqf`: means the rectier has paused on (اركب)": "Pelumeran dan pengucapan jelas dalam ayat {اركب معنا} dalam Surah Hud. Ini mengacu pada aturan bacaan tentang apakah huruf \"Nun\" (ن) dilumerkan ke huruf berikutnya atau diucapkan dengan jelas saat membaca ayat ini secara khusus. `waqf`: artinya pembaca telah berhenti pada (اركب)",
            "The nasalization (`ishmam`) or the slight drawing (`rawm`) in the verse {لا تأمنا على يوسف}": "Ghunnah (`ishmam`) atau penarikan sedikit (`rawm`) dalam ayat {لا تأمنا على يوسف}",
            "The vowel movement of the letter 'Dhad' (ض) (whether with `fath` or `dam`) in the word {ضعف} in Surah Ar-Rum.": "Gerakan vokal huruf 'Dhad' (ض) (apakah dengan `fathah` atau `dhammah`) dalam kata {ضعف} dalam Surah Ar-Rum.",
            "Affirmation and omission of the 'Alif' when pausing in the verse {سلاسلا} in Surah Al-Insan.This refers to the recitation rule regarding whether the final \"Alif\" in the word \"سلاسلا\" is pronounced (affirmed) or omitted when pausing (waqf) at this word during recitation in the specific verse from Surah Al-Insan. `hadhf`: means to remove alif (ا) during puase as (سلاسل) `ithbat`: means to recite alif (ا) during puase as (سلاسلا) `wasl` means completing the recitation as normal without pausing, so recite it as (سلاسلَ وأغلالا)": "Pengakuan dan pengabaian 'Alif' saat berhenti dalam ayat {سلاسلا} dalam Surah Al-Insan. Ini mengacu pada aturan bacaan tentang apakah \"Alif\" terakhir dalam kata \"سلاسلا\" diucapkan (dikonfirmasi) atau diabaikan saat berhenti (waqf) di kata ini selama bacaan dalam ayat spesifik dari Surah Al-Insan. `hadhf`: artinya menghapus alif (ا) saat berhenti seperti (سلاسل) `ithbat`: artinya membaca alif (ا) saat berhenti seperti (سلاسلا) `wasl` artinya menyelesaikan bacaan secara normal tanpa berhenti, jadi bacalah sebagai (سلاسلَ وأغلالا)",
            "Assimilation of the letter 'Qaf' into the letter 'Kaf,' whether incomplete (`idgham_naqis`) or complete (`idgham_kamil`), in the verse {نخلقكم} in Surah Al-Mursalat.": "Pelumeran huruf 'Qaf' ke huruf 'Kaf,' apakah tidak lengkap (`idgham_naqis`) atau lengkap (`idgham_kamil`), dalam ayat {نخلقكم} dalam Surah Al-Mursalat.",
            "Emphasis and softening of the letter 'Ra' in the word {urfq} in Surah Ash-Shu'ara' when connected (wasl).This refers to the recitation rules concerning whether the letter \"Ra\" (ر) in the word \"urfq\"  is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the specific verse from Surah Ash-Shu'ara' in connected speech. `waqf`: means pasuing so we only have one way (tafkheem of Raa)": "Penekanan dan pelunakan huruf 'Ra' dalam kata {urfq} dalam Surah Ash-Shu'ara' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf \"Ra\" (ر) dalam kata \"urfq\" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat membaca ayat spesifik dari Surah Ash-Shu'ara' secara bacaan bersambung. `waqf`: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)",
            'Emphasis and softening of the letter \'Ra\' in the word {urfq} in Surah Ash-Shu\'ara\' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "urfq"  is pronounced with emphasis (<code>tafkheem</code>) or softening (<code>tarqeeq</code>) when reciting the specific verse from Surah Ash-Shu\'ara\' in connected speech. <code>waqf</code>: means pasuing so we only have one way (tafkheem of Raa)': 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {urfq} dalam Surah Ash-Shu\'ara\' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "urfq" diucapkan dengan penekanan (<code>tafkheem</code>) atau pelunakan (<code>tarqeeq</code>) saat membaca ayat spesifik dari Surah Ash-Shu\'ara\' secara bacaan bersambung. <code>waqf</code>: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)',
            "Emphasis and softening of the letter 'Ra' in the word {القطر} in Surah Saba' when pausing (waqf).This refers to the recitation rules regarding whether the letter \"Ra\" (ر) in the word \"القطر\" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Saba'. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)": "Penekanan dan pelunakan huruf 'Ra' dalam kata {القطر} dalam Surah Saba' saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf \"Ra\" (ر) dalam kata \"القطر\" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Saba'. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)",
            "Emphasis and softening of the letter 'Ra' in the word {مصر} in Surah Yunus, and in the locations of Surah Yusuf and Surah Az-Zukhruf when pausing (waqf).This refers to the recitation rules regarding whether the letter \"Ra\" (ر) in the word \"مصر\" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) at the specific pauses in these Surahs. `wasl`: means not pasuing so we only have one way (tafkheem of Raa)": "Penekanan dan pelunakan huruf 'Ra' dalam kata {مصر} dalam Surah Yunus, dan dalam lokasi Surah Yusuf dan Surah Az-Zukhruf saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf \"Ra\" (ر) dalam kata \"مصر\" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) pada posisi berhenti spesifik dalam Surah-surah ini. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)",
            "Emphasis and softening of the letter 'Ra' in the word {نذر} in Surah Al-Qamar when pausing (waqf).This refers to the recitation rules regarding whether the letter \"Ra\" (ر) in the word \"نذر\" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Al-Qamar. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)": "Penekanan dan pelunakan huruf 'Ra' dalam kata {نذر} dalam Surah Al-Qamar saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf \"Ra\" (ر) dalam kata \"نذر\" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Al-Qamar. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)",
            "Emphasis and softening of the letter 'Ra' in the word {يسر} in Surah Al-Fajr when pausing (waqf).This refers to the recitation rules regarding whether the letter \"Ra\" (ر) in the word \"يسر\" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Al-Fajr. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)": "Penekanan dan pelunakan huruf 'Ra' dalam kata {يسر} dalam Surah Al-Fajr saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf \"Ra\" (ر) dalam kata \"يسر\" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Al-Fajr. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)",
            "This is not a standrad Hafs way but a disagreement between schoolars in our century how to pronounc Ikhfaa for meem. Some schoolars do full merging `إدام` and the other open the leaps a little bit `إخفاء`. We did not want to add this but some of the best reciters disagree about this": "Ini bukan cara standar Hafs tetapi perbedaan pendapat antara ulama di abad kita bagaimana mengucapkan Ikhfaa untuk meem. Beberapa ulama melakukan pelumeran penuh `إدام` dan yang lain membuka celah sedikit `إخفاء`. Kita tidak ingin menambahkan ini tetapi beberapa pembaca terbaik memiliki perbedaan pendapat tentang hal ini",
        }

        # Cari dan terapkan terjemahan yang cocok
        translated_help = help_translations.get(help, help)  # Gunakan teks terjemahan jika ada, jika tidak tetap teks asli

        # Jika teks bantuan belum diterjemahkan dan muncul dalam HTML Anda, coba terjemahkan
        if translated_help == help:
            # Tambahkan terjemahan spesifik untuk teks yang muncul dalam HTML
            if help == "The type of the quran Rewaya.":
                translated_help = "Jenis qira'ah Al-Quran."
            elif help == 'The ways to add takbeer (الله أكبر) after Istiaatha (استعاذة) and between end of the surah and beginning of the surah. <code>no_takbeer</code>: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah':
                translated_help = 'Cara menambahkan takbir (الله أكبر) setelah Istiaatha (استعاذة) dan antara akhir surah dan awal surah. <code>no_takbeer</code>: "لا تكبير" — Tanpa Takbir (Tanpa pengucapan kebesaran, yaitu tidak ada bacaan Takbir) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbir dari awal Surah Ash-Sharh ke awal Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbir dari akhir Surah Ad-Duha ke akhir Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbir di awal setiap Surah kecuali Surah At-Tawbah'
            elif help == ' The length of Mad Al Monfasel "مد النفصل" for Hafs Rewaya.':
                translated_help = 'Panjang Mad Al Monfasel (مد النفصل) untuk qira\'ah Hafs.'
            elif help == ' The length of Mad Al Motasel "مد المتصل" for Hafs.':
                translated_help = 'Panjang Mad Al Muttasil (مد المتصل) untuk qira\'ah Hafs.'
            elif help == ' The length of Madd Almotasel at pause for Hafs. Example "السماء".':
                translated_help = 'Panjang Mad Muttasil pada akhir ayat untuk qira\'ah Hafs. Contoh "السماء".'
            elif help == ' The length of Mad Al Aared "مد العارض للسكون".':
                translated_help = 'Panjang Mad Al Aared (مد العارض للسكون) untuk qira\'ah Hafs.'
            elif help == "The length of the Madd al-Leen when stopping at the end of a word (for a sakin waw or ya preceded by a letter with a fatha) should be less than or equal to the length of Madd al-'Arid (the temporary stretch due to stopping). **Default Value is equal to `madd_aared_len`**. مقدرا مع اللين عن القوف (للواو الساكنة والياء الساكنة وقبلها حرف مفتوح) ويجب أن يكون مقدار مد اللين أقل من أو يساوي مع العارض":
                translated_help = "Panjang Mad Al-Leen saat berhenti di akhir kata (untuk wawu sukun atau ya' yang didahului oleh huruf dengan fathah) seharusnya kurang dari atau sama dengan panjang Mad Al-Arid (pemanjangan sementara karena berhenti). **Nilai bawaan sama dengan `madd_aared_len`**. Nilai Mad Lin saat berhenti (untuk wawu sukun atau ya sukun dengan huruf fathah sebelumnya) dan nilai Mad Lin harus kurang dari atau sama dengan Mad Aared."
            elif help == "The ghonna for merging (Idghaam) noon with Lam and Raa for Hafs.":
                translated_help = "Ghunnah untuk menggabung (Idgham) nun dengan Lam dan Ra untuk Hafs."
            elif help == "The ways to recite the word meem Aal Imran (الم الله) at connected recitation. `waqf`: Pause with a prolonged madd (elongation) of 6 harakat (beats). `wasl_2` Pronounce \"meem\" with fathah (a short \"a\" sound) and stretch it for 2 harakat. `wasl_6` Pronounce \"meem\" with fathah and stretch it for 6 harakat.":
                translated_help = "Cara membaca kata meem Aal Imran (الم الله) pada bacaan bersambung. `waqf`: Berhenti dengan mad panjang (pemanjangan) selama 6 harakat (ketukan). `wasl_2` Ucapkan \"meem\" dengan fathah (suara pendek \"a\") dan ulurkan selama 2 harakat. `wasl_6` Ucapkan \"meem\" dengan fathah dan ulurkan selama 6 harakat."
            elif help == " The length of Lzem Harfy of Yaa in letter Al-Ayen Madd \"المد الحرفي اللازم لحرف العين\" in surar: Maryam \"مريم\", AlShura \"الشورى\".":
                translated_help = "Panjang Lzem Harfy dari Ya dalam huruf Al-Ayen Madd \"المد الحرفي اللازم لحرف العين\" dalam surah: Maryam \"مريم\", AlShura \"الشورى\"."
            elif help == "The ways of Hafs for saken before hamz. \"The letter with sukoon before the hamzah (ء)\".And it has three forms: full articulation (`tahqeeq`), general pause (`general_sakt`), and specific pause (`local_skat`).":
                translated_help = "Cara Hafs untuk sakt sebelum hamz. \"Huruf dengan sukun sebelum hamzah (ء)\".Dan memiliki tiga bentuk: penuh (`tahqeeq`), jeda umum (`general_sakt`), dan jeda khusus (`local_skat`)."
            elif help == 'The ways to recite the word "عوجا" (Iwaja). `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).':
                translated_help = 'Cara membaca kata "عوجا" (Iwaja). `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).'
            elif help == 'The ways to recite the word "مرقدنا" (Marqadena) in Surat Yassen. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).':
                translated_help = 'Cara membaca kata "مرقدنا" (Marqadena) dalam Surat Yasin. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).'
            elif help == 'The ways to recite the word "من راق" (Man Raq) in Surat Al Qiyama. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).':
                translated_help = 'Cara membaca kata "من راق" (Man Raq) dalam Surat Al Qiyama. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).'
            elif help == 'The ways to recite the word "بل ران" (Bal Ran) in Surat Al Motaffin. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).':
                translated_help = 'Cara membaca kata "بل ران" (Bal Ran) dalam Surat Al Motaffin. `sakt` artinya jeda sedikit. `idraj` artinya bukan `sakt`. `waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idraj` (tidak bersakt).'
            elif help == 'The ways to recite the word {ماليه هلك} in Surah Al-Ahqaf. `sakt` means slight pause. `idgham` Assimilation of the letter \'Ha\' (ه) into the letter \'Ha\' (ه) with complete assimilation.`waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idgham`.':
                translated_help = 'Cara membaca kata {ماليه هلك} dalam Surah Al-Ahqaf. `sakt` artinya jeda sedikit. `idgham` Pelumeran huruf \'Ha\' (ه) ke huruf \'Ha\' (ه) dengan pelumeran lengkap.`waqf`: artinya jeda penuh, jadi kita tidak bisa menentukan apakah pembaca menggunakan `sakt` atau `idgham`.'
            elif help == 'The ways to recite end of Surah Al-Anfal and beginning of Surah At-Tawbah.':
                translated_help = 'Cara membaca akhir Surah Al-Anfal dan awal Surah At-Tawbah.'
            elif help == 'Weither to merge noon of both: {يس} and {ن} with (و) "`idgham`" or not "`izhar`".':
                translated_help = 'Apakah menggabungkan nun dari kedua: {يس} dan {ن} dengan (و) "`idgham`" atau tidak "`izhar`".'
            elif help == "The affirmation and omission of the letter 'Yaa' in the pause of the verse {آتاني} in Surah An-Naml.`wasl`: means connected recitation without pasuding as (آتانيَ).`hadhf`: means deletion of letter (ي) at puase so recited as (آتان).`ithbat`: means confirmation reciting letter (ي) at puase as (آتاني).":
                translated_help = "Pengakuan dan pengabaian huruf 'Yaa' dalam jeda ayat {آتاني} dalam Surah An-Naml.`wasl`: artinya bacaan bersambung tanpa jeda seperti (آتانيَ).`hadhf`: artinya penghapusan huruf (ي) saat jeda jadi dibaca (آتان).`ithbat`: artinya konfirmasi membaca huruf (ي) saat jeda seperti (آتاني)."
            elif help == "The ruling on starting with the word {الاسم} in Surah Al-Hujurat.`lism` Recited as (لسم) at the beginning. `alism` Recited as (ألسم). ath the beginning`wasl`: means completing recitaion without paussing as normal, So Reciting is as (بئس لسم).":
                translated_help = "Hukum memulai dengan kata {الاسم} dalam Surah Al-Hujurat.`lism` Dibaca sebagai (لسم) di awal. `alism` Dibaca sebagai (ألسم). di awal`wasl`: artinya menyelesaikan bacaan tanpa jeda seperti biasa, Jadi dibaca sebagai (بئس لسم)."
            elif help == 'The ruling on pronouncing `seen` (س) or `saad` (ص) in the verse {والله يقبض ويبسط} in Surah Al-Baqarah.':
                translated_help = 'Hukum mengucapkan `sin` (س) atau `shad` (ص) dalam ayat {والله يقبض ويبسط} dalam Surah Al-Baqarah.'
            elif help == 'The ruling on pronouncing `seen` (س) or `saad` (ص ) in the verse {وزادكم في الخلق بسطة} in Surah Al-A\'raf.':
                translated_help = 'Hukum mengucapkan `sin` (س) atau `shad` (ص) dalam ayat {وزادكم في الخلق بسطة} dalam Surah Al-A\'raf.'
            elif help == 'The pronunciation of `seen` (س) or `saad` (ص ) in the verse {أم هم المصيطرون} in Surah At-Tur.':
                translated_help = 'Pengucapan `sin` (س) atau `shad` (ص) dalam ayat {أم هم المصيطرون} dalam Surah At-Tur.'
            elif help == 'The pronunciation of `seen` (س) or `saad` (ص ) in the verse {لست عليهم بمصيطر} in Surah Al-Ghashiyah.':
                translated_help = 'Pengucapan `sin` (س) atau `shad` (ص) dalam ayat {لست عليهم بمصيطر} dalam Surah Al-Ghashiyah.'
            elif help == ' Tasheel of Madd "وجع التسهيل أو المد" for 6 words in The Holy Quran: "ءالذكرين", "ءالله", "ءائن".':
                translated_help = 'Tashil dari Mad "وجع التسهيل أو المد" untuk 6 kata dalam Al-Quran Mulia: "ءالذكرين", "ءالله", "ءائن".'
            elif help == 'The assimilation (`idgham`) and non-assimilation (`izhar`) in the verse {يلهث ذلك} in Surah Al-A\'raf. `waqf`: means the rectier has paused on (يلهث)':
                translated_help = 'Pelumeran (`idgham`) dan bukan pelumeran (`izhar`) dalam ayat {يلهث ذلك} dalam Surah Al-A\'raf. `waqf`: artinya pembaca telah berhenti pada (يلهث)'
            elif help == 'The assimilation and clear pronunciation in the verse {اركب معنا} in Surah Hud.This refers to the recitation rules concerning whether the letter "Noon" (ن) is assimilated into the following letter or pronounced clearly when reciting this specific verse. `waqf`: means the rectier has paused on (اركب)':
                translated_help = 'Pelumeran dan pengucapan jelas dalam ayat {اركب معنا} dalam Surah Hud. Ini mengacu pada aturan bacaan tentang apakah huruf "Nun" (ن) dilumerkan ke huruf berikutnya atau diucapkan dengan jelas saat membaca ayat ini secara khusus. `waqf`: artinya pembaca telah berhenti pada (اركب)'
            elif help == 'The nasalization (`ishmam`) or the slight drawing (`rawm`) in the verse {لا تأمنا على يوسف}':
                translated_help = 'Ghunnah (`ishmam`) atau penarikan sedikit (`rawm`) dalam ayat {لا تأمنا على يوسف}'
            elif help == 'The vowel movement of the letter \'Dhad\' (ض) (whether with `fath` or `dam`) in the word {ضعف} in Surah Ar-Rum.':
                translated_help = 'Gerakan vokal huruf \'Dhad\' (ض) (apakah dengan `fathah` atau `dhammah`) dalam kata {ضعف} dalam Surah Ar-Rum.'
            elif help == 'Affirmation and omission of the \'Alif\' when pausing in the verse {سلاسلا} in Surah Al-Insan.This refers to the recitation rule regarding whether the final "Alif" in the word "سلاسلا" is pronounced (affirmed) or omitted when pausing (waqf) at this word during recitation in the specific verse from Surah Al-Insan. `hadhf`: means to remove alif (ا) during puase as (سلاسل) `ithbat`: means to recite alif (ا) during puase as (سلاسلا) `wasl` means completing the recitation as normal without pausing, so recite it as (سلاسلَ وأغلالا)':
                translated_help = 'Pengakuan dan pengabaian \'Alif\' saat berhenti dalam ayat {سلاسلا} dalam Surah Al-Insan. Ini mengacu pada aturan bacaan tentang apakah "Alif" terakhir dalam kata "سلاسلا" diucapkan (dikonfirmasi) atau diabaikan saat berhenti (waqf) di kata ini selama bacaan dalam ayat spesifik dari Surah Al-Insan. `hadhf`: artinya menghapus alif (ا) saat berhenti seperti (سلاسل) `ithbat`: artinya membaca alif (ا) saat berhenti seperti (سلاسلا) `wasl` artinya menyelesaikan bacaan secara normal tanpa berhenti, jadi bacalah sebagai (سلاسلَ وأغلالا)'
            elif help == 'Assimilation of the letter \'Qaf\' into the letter \'Kaf,\' whether incomplete (`idgham_naqis`) or complete (`idgham_kamil`), in the verse {نخلقكم} in Surah Al-Mursalat.':
                translated_help = 'Pelumeran huruf \'Qaf\' ke huruf \'Kaf\', apakah tidak lengkap (`idgham_naqis`) atau lengkap (`idgham_kamil`), dalam ayat {نخلقكم} dalam Surah Al-Mursalat.'
            elif help == 'Emphasis and softening of the letter \'Ra\' in the word {urfq} in Surah Ash-Shu\'ara\' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "urfq"  is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the specific verse from Surah Ash-Shu\'ara\' in connected speech. `waqf`: means pasuing so we only have one way (tafkheem of Raa)':
                translated_help = 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {urfq} dalam Surah Ash-Shu\'ara\' saat bersambung (wasl). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "urfq" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat membaca ayat spesifik dari Surah Ash-Shu\'ara\' secara bacaan bersambung. `waqf`: artinya berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)'
            elif help == 'Emphasis and softening of the letter \'Ra\' in the word {القطر} in Surah Saba\' when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "القطر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Saba\'. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)':
                translated_help = 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {القطر} dalam Surah Saba\' saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "القطر" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Saba\'. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)'
            elif help == 'Emphasis and softening of the letter \'Ra\' in the word {مصر} in Surah Yunus, and in the locations of Surah Yusuf and Surah Az-Zukhruf when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "مصر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) at the specific pauses in these Surahs. `wasl`: means not pasuing so we only have one way (tafkheem of Raa)':
                translated_help = 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {مصر} dalam Surah Yunus, dan dalam lokasi Surah Yusuf dan Surah Az-Zukhruf saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "مصر" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) pada posisi berhenti spesifik dalam Surah-surah ini. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tafkheem dari Ra)'
            elif help == 'Emphasis and softening of the letter \'Ra\' in the word {نذر} in Surah Al-Qamar when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "نذر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Al-Qamar. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)':
                translated_help = 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {نذر} dalam Surah Al-Qamar saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "نذر" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Al-Qamar. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)'
            elif help == 'Emphasis and softening of the letter \'Ra\' in the word {يسر} in Surah Al-Fajr when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "يسر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Al-Fajr. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)':
                translated_help = 'Penekanan dan pelunakan huruf \'Ra\' dalam kata {يسر} dalam Surah Al-Fajr saat berhenti (waqf). Ini mengacu pada aturan bacaan tentang apakah huruf "Ra" (ر) dalam kata "يسر" diucapkan dengan penekanan (`tafkheem`) atau pelunakan (`tarqeeq`) saat berhenti di kata ini dalam Surah Al-Fajr. `wasl`: artinya tidak berhenti jadi kita hanya punya satu cara (tarqeeq dari Ra)'
            elif help == 'This is not a standrad Hafs way but a disagreement between schoolars in our century how to pronounc Ikhfaa for meem. Some schoolars do full merging `إدام` and the other open the leaps a little bit `إخفاء`. We did not want to add this but some of the best reciters disagree about this':
                translated_help = 'Ini bukan cara standar Hafs tetapi perbedaan pendapat antara ulama di abad kita bagaimana mengucapkan Ikhfaa untuk meem. Beberapa ulama melakukan pelumeran penuh `إ DAM` dan yang lain membuka celah sedikit `إخفاء`. Kita tidak ingin menambahkan ini tetapi beberapa pembaca terbaik memiliki perbedaan pendapat tentang hal ini'
            elif help == 'The length of Madd Almotasel at pause for Hafs.. Example "السماء".':
                translated_help = 'Panjang Mad Muttasil pada akhir ayat untuk qira\'ah Hafs. Contoh "السماء".'
            elif help == 'The length of Madd Almotasel at pause for Hafs.. Example "السماء".':  # Duplikat - untuk keamanan
                translated_help = 'Panjang Mad Muttasil pada akhir ayat untuk qira\'ah Hafs. Contoh "السماء".'
            elif help == 'The length of Mad Al Motasel "مد المتصل" for Hafs.':
                translated_help = 'Panjang Mad Al Muttasil (مد المتصل) untuk qira\'ah Hafs.'
            elif help == 'The ways to recite the word meem Aal Imran (الم الله) at connected recitation. `waqf`: Pause with a prolonged madd (elongation) of 6 harakat (beats). `wasl_2` Pronounce "meem" with fathah (a short "a" sound) and stretch it for 2 harakat. `wasl_6` Pronounce "meem" with fathah and stretch it for 6 harakat.':
                translated_help = 'Cara membaca kata meem Aal Imran (الم الله) pada bacaan bersambung. `waqf`: Berhenti dengan mad panjang (pemanjangan) selama 6 harakat (ketukan). `wasl_2` Ucapkan "meem" dengan fathah (suara pendek "a") dan ulurkan selama 2 harakat. `wasl_6` Ucapkan "meem" dengan fathah dan ulurkan selama 6 harakat.'
            elif help == 'The ways to add takbeer (الله أكبر) after Istiaatha (استعاذة) and between end of the surah and beginning of the surah. <code>no_takbeer</code>: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah':
                translated_help = 'Cara menambahkan takbir (الله أكبر) setelah Istiaatha (استعاذة) dan antara akhir surah dan awal surah. <code>no_takbeer</code>: "لا تكبير" — Tanpa Takbir (Tanpa pengucapan kebesaran, yaitu tidak ada bacaan Takbir) <code>beginning_of_sharh</code>: "التكبير من أول الشرح لأول الناس" — Takbir dari awal Surah Ash-Sharh ke awal Surah An-Nas <code>end_of_dohaf</code>: "التكبير من آخر الضحى لآخر الناس" — Takbir dari akhir Surah Ad-Duha ke akhir Surah An-Nas <code>general_takbeer</code>: "التكبير أول كل سورة إلا التوبة" — Takbir di awal setiap Surah kecuali Surah At-Tawbah'

        help = translated_help

    # Handle Literal types
    if get_origin(field_info.annotation) is Literal:
        choices = list(get_args(field_info.annotation))
        arabic_attributes = get_arabic_attributes(field_info)

        # Create choice list with Arabic labels if available
        choice_list = []
        for choice in choices:
            if arabic_attributes and choice in arabic_attributes:
                choice_list.append((arabic_attributes[choice], choice))
            else:
                choice_list.append((str(choice), choice))

        return gr.Dropdown(
            choices=choice_list,
            value=default_value,
            label=label,
            info=help,
            interactive=True,
        )

    # Handle different field types
    if field_info.annotation in [str, Optional[str]]:
        return gr.Textbox(value=default_value or "", label=label, info=help)
    elif field_info.annotation in [int, Optional[int]]:
        return gr.Number(value=default_value or 0, label=label, info=help, precision=0)
    elif field_info.annotation in [float, Optional[float]]:
        return gr.Number(
            value=default_value or 0.0, label=label, info=help, precision=1
        )
    elif field_info.annotation in [bool, Optional[bool]]:
        return gr.Checkbox(value=default_value or False, label=label, info=help)

    raise ValueError(f"Unsupported field type for {label}: {field_info.annotation}")


def update_aya_dropdown(sura_idx):
    if not sura_idx:
        sura_idx = 1
    return gr.update(
        choices=list(range(1, sura_to_aya_count[int(sura_idx)] + 1)), value=1
    )


def update_uthmani_ref(sura_idx, aya_idx, start_idx, num_words):
    if not all([sura_idx, aya_idx, start_idx is not None, num_words is not None]):
        return ""
    try:
        uthmani_ref = (
            Aya(int(sura_idx), int(aya_idx))
            .get_by_imlaey_words(int(start_idx), int(num_words))
            .uthmani
        )
        return uthmani_ref
    except PartOfUthmaniWord as e:
        return f"⚠️ Peringatan: Anda telah memilih sebagian kata Utsmani. Silakan sesuaikan jumlah kata untuk hanya mencakup kata lengkap.\n\nDetail kesalahan: {str(e)}"
    except Exception as e:
        return f"Kesalahan: {str(e)}"


def process_audio(audio, sura_idx, aya_idx, start_idx, num_words):
    global current_moshaf

    if audio is None:
        return "Silakan unggah file audio terlebih dahulu"

    try:
        # Get Uthmani reference text
        uthmani_ref = (
            Aya(int(sura_idx), int(aya_idx))
            .get_by_imlaey_words(int(start_idx), int(num_words))
            .uthmani
        )
        phonetizer_out = quran_phonetizer(
            uthmani_ref, current_moshaf, remove_spaces=True
        )

        # Process audio
        wave, _ = load(audio, sr=sampling_rate, mono=True)
        outs = muaalem(
            [wave],
            [phonetizer_out],
            sampling_rate=sampling_rate,
        )

        # # Prepare output
        # output_text = f"Phonemes: {outs[0].phonemes}\n\n"
        # for sifa in outs[0].sifat:
        #     output_text += json.dumps(asdict(sifa), indent=2, ensure_ascii=False) + "\n"
        #     output_text += "*" * 30 + "\n"
        # output_text += "-" * 40 + "\n\n"

        # Add explanation
        explanation_html = explain_for_gradio(
            outs[0].phonemes.text,
            phonetizer_out.phonemes,
            outs[0].sifat,
            phonetizer_out.sifat,
            lang="arabic",
        )

        return explanation_html

    except PartOfUthmaniWord as e:
        return f"⚠️ Peringatan: Rentang kata yang dipilih mencakup kata Utsmani sebagian. Silakan sesuaikan jumlah kata untuk hanya mencakup kata lengkap.\n\nDetail kesalahan: {str(e)}"
    # except Exception as e:
    #     return f"Kesalahan memproses audio: {str(e)}"


def update_moshaf_settings(*args):
    """Update the global moshaf settings with values from the settings page"""
    global current_moshaf, field_names

    try:
        # Create a dictionary from the field names and values
        settings_dict = dict(zip(field_names, args))

        # Create a new MoshafAttributes object with the updated values
        current_moshaf = MoshafAttributes(**settings_dict)
        return "✅ Pengaturan berhasil disimpan - Settings saved successfully!"
    except Exception as e:
        return f"❌ Kesalahan saat menyimpan pengaturan - Error saving settings: {str(e)}"


def reset_settings():
    """Reset all settings to default values"""
    global current_moshaf

    try:
        current_moshaf = default_moshaf
        # Return default values for all fields
        default_values = [
            getattr(default_moshaf, field_name) for field_name in field_names
        ]
        return default_values + [
            "✅ Berhasil mengembalikan ke pengaturan awal - Reset to default settings successfully!"
        ]
    except Exception as e:
        return [getattr(current_moshaf, field_name) for field_name in field_names] + [
            f"❌ Error resetting settings: {str(e)}"
        ]


# Create the Gradio app
with gr.Blocks(title="Pengajar Al-Quran") as app:
    # Store current moshaf settings in session state
    current_moshaf_state = gr.State(default_moshaf)

    # Initialize field names list
    field_names = []

    with gr.Tab("Analisis Utama - Main Analysis"):
        gr.Markdown("# Deteksi Kesalahan Bacaan, Tajwid, dan Sifat Huruf")
        gr.Markdown("Pilih bagian Al-Quran yang ingin Anda pelajari")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Perbandingan Bacaan")

                # Create sura dropdown with both index and name
                sura_choices = [
                    (f"{idx} - {sura_idx_to_name[idx]}", idx) for idx in range(1, 115)
                ]
                sura_dropdown = gr.Dropdown(
                    choices=sura_choices,
                    label="Surah",
                    value=1,
                    elem_id="sura_dropdown",
                )

                aya_dropdown = gr.Dropdown(
                    choices=list(range(1, sura_to_aya_count[1] + 1)),
                    label="Nomor Ayat",
                    value=1,
                    elem_id="aya_dropdown",
                )
                start_idx = gr.Number(
                    value=0,
                    label="Nomor kata dimulai dari nol (Word Index)",
                    minimum=0,
                    step=1,
                    elem_id="start_idx",
                )
                num_words = gr.Number(
                    value=4,
                    label="Jumlah kata",
                    minimum=1,
                    step=1,
                    elem_id="num_words",
                )
                uthmani_text = gr.Textbox(
                    label="Tulisan Utsmani",
                    interactive=False,
                    elem_id="uthmani_text",
                )

            with gr.Column(scale=2):
                gr.Markdown("### فحص التلاوة القرآنية")
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    label="Unggah atau Rekam Audio",
                    type="filepath",
                    elem_id="audio_input",
                )
                analyze_btn = gr.Button(
                    "Periksa Bacaan", variant="primary", elem_id="analyze_btn"
                )
                output_html = gr.HTML(
                    label="Hasil pemeriksaan",
                    elem_id="output_html",
                )

        # Initial update of uthmani text
        app.load(
            update_uthmani_ref,
            inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=uthmani_text,
        )

        # Update aya dropdown when sura changes and reset aya_idx to 1
        sura_dropdown.change(
            update_aya_dropdown, inputs=sura_dropdown, outputs=aya_dropdown
        ).then(
            update_uthmani_ref,
            inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=uthmani_text,
        )

        # Update uthmani text when any parameter changes
        for component in [aya_dropdown, start_idx, num_words]:
            component.change(
                update_uthmani_ref,
                inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
                outputs=uthmani_text,
            )

        # Process audio when button is clicked
        analyze_btn.click(
            process_audio,
            inputs=[audio_input, sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=output_html,
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
            update_moshaf_settings, inputs=settings_components, outputs=status_message
        )

        # Reset to default event
        reset_btn.click(
            reset_settings, inputs=[], outputs=settings_components + [status_message]
        )


def main(app=app):
#    app.launch(server_name="0.0.0.0", share=False)
    app.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()
    # app.launch(server_name="0.0.0.0", share=True)
