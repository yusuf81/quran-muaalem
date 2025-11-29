"""Configuration module for Quran Muaalem."""

from .constants import (
    REQUIRED_MOSHAF_FIELDS,
    device,
    sampling_rate,
    sura_idx_to_name,
    sura_to_aya_count,
    default_moshaf,
)

__all__ = [
    'REQUIRED_MOSHAF_FIELDS',
    'device',
    'sampling_rate',
    'sura_idx_to_name',
    'sura_to_aya_count',
    'default_moshaf',
]
