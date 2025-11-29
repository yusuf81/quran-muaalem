"""Constants and configuration for Quran Muaalem application."""

import torch
from quran_transcript import Aya, MoshafAttributes


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Audio configuration
sampling_rate = 16000

# Required Moshaf fields for settings UI
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

# Default moshaf settings
default_moshaf = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=4,
)

# Load Sura information
sura_idx_to_name = {}
sura_to_aya_count = {}

def _initialize_sura_info():
    """Initialize sura name and ayah count mappings."""
    start_aya = Aya()
    for sura_idx in range(1, 115):
        start_aya.set(sura_idx, 1)
        sura_idx_to_name[sura_idx] = start_aya.get().sura_name
        sura_to_aya_count[sura_idx] = start_aya.get().num_ayat_in_sura

# Initialize on module import
_initialize_sura_info()
