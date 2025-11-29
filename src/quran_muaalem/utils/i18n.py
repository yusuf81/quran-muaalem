"""Translation manager for multi-language support."""

import json
from pathlib import Path
from typing import Optional


class TranslationManager:
    """Manage translations for the application."""
    
    def __init__(self, locale: str = 'id'):
        """Initialize translation manager.
        
        Args:
            locale: Language code (default: 'id' for Indonesian)
        """
        self.locale = locale
        self.translations = self._load_translations()
    
    def _load_translations(self) -> dict:
        """Load translations from JSON file."""
        trans_file = Path(__file__).parent.parent.parent / f'translations/{self.locale}.json'
        
        if not trans_file.exists():
            # Fallback to empty dict if file doesn't exist
            return {"field_labels": {}, "field_descriptions": {}}
        
        with open(trans_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_field_label(self, field_name: str, default: Optional[str] = None) -> str:
        """Get translated label for a field.
        
        Args:
            field_name: Name of the field
            default: Default value if translation not found
            
        Returns:
            Translated label or default value
        """
        return self.translations.get('field_labels', {}).get(
            field_name, 
            default or field_name
        )
    
    def get_field_description(self, field_name: str, default: Optional[str] = None) -> str:
        """Get translated description for a field.
        
        Args:
            field_name: Name of the field
            default: Default value if translation not found
            
        Returns:
            Translated description or default value
        """
        return self.translations.get('field_descriptions', {}).get(
            field_name,
            default or ""
        )


# Global instance
_translator = None


def get_translator(locale: str = 'id') -> TranslationManager:
    """Get translation manager instance.
    
    Args:
        locale: Language code
        
    Returns:
        TranslationManager instance
    """
    global _translator
    if _translator is None or _translator.locale != locale:
        _translator = TranslationManager(locale)
    return _translator
