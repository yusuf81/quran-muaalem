# Kritik Repository Quran Muaalem

**Tanggal Evaluasi:** 29 November 2025  
**Versi yang Dievaluasi:** Fork oleh yusuf81 dari obadx/quran-muaalem

---

## 📋 Ringkasan Eksekutif

Repository ini adalah fork dari proyek akademik yang solid untuk deteksi kesalahan tajwid Al-Quran menggunakan deep learning. Modifikasi yang ditambahkan (translasi Indonesia, segmentasi multi-ayat) menambah nilai untuk pengguna lokal. Namun, ada beberapa area kritis yang memerlukan perbaikan terkait kualitas kode, maintainability, dan best practices pengembangan software.

**Skor Keseluruhan: 6.5/10**

---

## 🔴 Masalah Kritis (High Priority)

### 1. **File Gradio App yang Terlalu Besar (1089 baris)**

**Masalah:**
- File `gradio_app.py` berisi 1089 baris kode dalam satu file
- Mencampur logika UI, business logic, dan data processing
- Sulit untuk maintain, test, dan debug
- Melanggar prinsip Single Responsibility Principle (SRP)

**Lokasi:** `src/quran_muaalem/gradio_app.py`

**Dampak:**
- Sulit untuk menulis unit tests yang terpisah
- Perubahan kecil berisiko break functionality lain
- Code review menjadi sangat sulit
- Onboarding developer baru akan lambat

**Rekomendasi:**
```
Refactor menjadi struktur modular:
src/quran_muaalem/
  ├── ui/
  │   ├── __init__.py
  │   ├── components.py      # Gradio components
  │   ├── handlers.py         # Event handlers
  │   └── layouts.py          # UI layouts
  ├── services/
  │   ├── audio_processor.py  # Audio processing logic
  │   ├── analysis_service.py # Analysis logic
  │   └── segmentation.py     # Segmentation logic
  └── utils/
      ├── translations.py     # Translation mappings
      └── constants.py        # Constants & configs
```

### 2. **Hardcoded Translation Dictionary (400+ baris)**

**Masalah:**
- Dictionary terjemahan hardcoded di dalam fungsi `create_gradio_input_for_field()` (baris 228-366)
- 400+ baris dictionary yang sangat sulit dibaca dan maintain
- Duplikasi string yang sama berulang kali
- Tidak scalable untuk penambahan bahasa

**Lokasi:** `gradio_app.py` baris 228-459

**Contoh dari kode:**
```python
help_translations = {
    "The Rewaya to use for recitation.": "Qiraah yang digunakan untuk bacaan.",
    # ... 100+ entri lainnya yang similar
    'Emphasis and softening of the letter \'Ra\' in the word {urfq}...': '...',
    # Duplikasi dengan variasi tag HTML
}
```

**Dampak:**
- Code bloat yang ekstrem
- Maintenance nightmare
- Bug prone (typo dalam string panjang)
- Impossible untuk manage translations secara terpisah

**Rekomendasi:**
```python
# translations/id.json
{
  "settings.rewaya.label": "Qiraah",
  "settings.rewaya.description": "Qiraah yang digunakan untuk bacaan.",
  ...
}

# utils/i18n.py
class TranslationManager:
    def __init__(self, locale='id'):
        self.translations = load_json(f'translations/{locale}.json')
    
    def t(self, key, default=None):
        return self.translations.get(key, default or key)
```

### 3. **Global Mutable State**

**Masalah:**
- Penggunaan global variables yang mutable: `current_moshaf`, `segmenter_model`, `segmenter_processor`
- State management yang tidak thread-safe
- Sulit untuk testing dan debugging

**Lokasi:** 
- `gradio_app.py` baris 156, 588-590
- `gradio_app.py` baris 81 (`muaalem` global instance)

**Contoh:**
```python
# Global mutable state - BAD PRACTICE
current_moshaf = default_moshaf  # Line 156

def update_moshaf_settings(*args):
    global current_moshaf  # Modifying global state
    current_moshaf = MoshafAttributes(**settings_dict)
```

**Dampak:**
- Race conditions pada concurrent requests
- State tidak predictable
- Testing memerlukan setup/teardown yang kompleks
- Debugging sangat sulit

**Rekomendasi:**
```python
# Gunakan Gradio State atau dependency injection
class AppState:
    def __init__(self):
        self.moshaf = default_moshaf
        self.segmenter = None
    
    def update_moshaf(self, **kwargs):
        self.moshaf = MoshafAttributes(**kwargs)
        return self.moshaf

# Dalam Gradio app
app_state = gr.State(AppState())
```

### 4. **Lazy Loading yang Tidak Konsisten**

**Masalah:**
- Segmenter di-load secara lazy (line 593-602)
- Main model `muaalem` di-load di global scope (line 81)
- Tidak ada error handling untuk model loading failure

**Lokasi:** `gradio_app.py` baris 81, 593-602

**Code:**
```python
# Main model loaded immediately - konsumsi memory bahkan sebelum digunakan
muaalem = Muaalem(model_name_or_path=model_id, device=device)  # Line 81

# Segmenter loaded lazily
def load_segmenter():
    global segmenter_model, segmenter_processor
    if segmenter_model is None:
        # Load model...
```

**Dampak:**
- Inconsistent behavior
- Startup time yang lama
- Memory waste jika user tidak menggunakan main analysis
- No graceful degradation

**Rekomendasi:**
```python
class ModelManager:
    def __init__(self):
        self._muaalem = None
        self._segmenter = None
    
    @property
    def muaalem(self):
        if self._muaalem is None:
            try:
                self._muaalem = Muaalem(...)
            except Exception as e:
                logger.error(f"Failed to load muaalem: {e}")
                raise
        return self._muaalem
    
    # Similar untuk segmenter
```

### 5. **Duplikasi Kode Build Directory**

**Masalah:**
- Direktori `build/` berisi duplikasi lengkap dari source code
- Tidak ada di `.gitignore`
- Membingungkan developer mana yang actual source

**Lokasi:** `/root/quran-muaalem/build/`

**Dampak:**
- Repository bloat
- Developer bisa salah edit file di `build/` instead of `src/`
- Confusion tentang source of truth
- Waste storage

**Rekomendasi:**
```gitignore
# Add to .gitignore
build/
dist/
*.egg-info/
```

### 6. **Error Handling yang Lemah**

**Masalah:**
- Broad exception catching tanpa specific handling
- Error messages yang tidak informatif
- Tidak ada logging yang proper

**Contoh dari `gradio_app.py`:
```python
except Exception as e:  # Too broad!
    return f"Kesalahan: {str(e)}"  # User tidak tahu apa yang harus dilakukan

try:
    # Complex operation
except PartOfUthmaniWord as e:  # Good - specific
    return (None, f"Kesalahan memproses audio: {str(e)}")  # Still not ideal
```

**Rekomendasi:**
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = process_audio(...)
except AudioProcessingError as e:
    logger.error(f"Audio processing failed: {e}", exc_info=True)
    return create_error_response(
        "Gagal memproses audio",
        details=str(e),
        suggestion="Pastikan file audio valid dan tidak corrupt"
    )
except ModelInferenceError as e:
    logger.error(f"Model inference failed: {e}", exc_info=True)
    return create_error_response(
        "Gagal melakukan analisis",
        suggestion="Coba lagi atau hubungi support"
    )
```

---

## 🟡 Masalah Menengah (Medium Priority)

### 7. **Tidak Ada CI/CD Pipeline**

**Masalah:**
- Tidak ada GitHub Actions atau CI/CD setup
- Testing tidak automated
- Tidak ada linting/formatting checks
- Deployment manual

**Rekomendasi:**
Tambahkan `.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .[test]
      - name: Run tests
        run: pytest tests/
      - name: Run linting
        run: |
          pip install ruff
          ruff check .
```

### 8. **Dokumentasi API yang Kurang**

**Masalah:**
- Docstrings ada tapi tidak konsisten
- Tidak ada type hints di semua function
- Tidak ada examples dalam docstrings
- README tidak menjelaskan modification secara detail

**Contoh dari kode:**
```python
def update_aya_dropdown(sura_idx):  # Missing type hints
    if not sura_idx:
        sura_idx = 1
    return gr.update(
        choices=list(range(1, sura_to_aya_count[int(sura_idx)] + 1)), value=1
    )
```

**Rekomendasi:**
```python
def update_aya_dropdown(sura_idx: int | None) -> gr.update:
    """Update ayah dropdown choices based on selected surah.
    
    Args:
        sura_idx: Index of the surah (1-114), or None to default to 1
        
    Returns:
        Gradio update object with new choices and default value
        
    Example:
        >>> update_aya_dropdown(2)  # Al-Baqarah has 286 verses
        gr.update(choices=[1, 2, ..., 286], value=1)
    """
    if not sura_idx:
        sura_idx = 1
    return gr.update(
        choices=list(range(1, sura_to_aya_count[int(sura_idx)] + 1)), 
        value=1
    )
```

### 9. **Test Coverage yang Tidak Lengkap**

**Masalah:**
- Tests hanya untuk core modules
- Tidak ada tests untuk UI logic
- Tidak ada integration tests
- Tidak ada tests untuk multi-verse segmentation (fitur baru)

**Files yang ada tests:**
- ✅ `decode.py` - Well tested
- ✅ `inference.py` - Basic test ada
- ❌ `gradio_app.py` - NO TESTS!
- ❌ `explain_gradio.py` - NO TESTS!
- ❌ Multi-verse segmentation - NO TESTS!

**Rekomendasi:**
```python
# tests/test_ui_logic.py
def test_toggle_aya_inputs_full_sura():
    """Test that full sura toggle disables aya inputs."""
    result = toggle_aya_inputs(is_full=True, sura_idx=2)
    assert result[mv_start_aya]['value'] == 1
    assert result[mv_end_aya]['value'] == 286  # Al-Baqarah
    assert result[mv_start_aya]['interactive'] == False

# tests/test_segmentation.py
def test_multi_verse_segmentation():
    """Test audio segmentation for multiple verses."""
    # Mock audio with clear pauses
    # Test segmentation accuracy
    # Test handling of mismatch between expected and detected
```

### 10. **Hardcoded Magic Numbers dan Strings**

**Masalah:**
- Magic numbers tersebar di kode
- No central configuration
- String literals di-duplicate

**Contoh:**
```python
# gradio_app.py
sampling_rate = 16000  # Line 82
pad_samples = int(0.5 * sampling_rate)  # Line 677
padding = sr  # 1 second * sample rate  # Line 114
min_silence_duration_ms=200,  # Line 650
min_speech_duration_ms=500,  # Line 651
```

**Rekomendasi:**
```python
# config.py
class AudioConfig:
    SAMPLING_RATE = 16000
    SEGMENT_PADDING_SECONDS = 0.5
    TRIM_PADDING_SECONDS = 1.0
    MIN_SILENCE_MS = 200
    MIN_SPEECH_MS = 500
    TRIM_DB_THRESHOLD = 25

class UIConfig:
    MAX_PREVIEW_VERSES = 20
    DEFAULT_SURA = 1
    DEFAULT_START_AYA = 1
    DEFAULT_END_AYA = 5
```

### 11. **Dependency Management yang Kurang Ketat**

**Masalah:**
- `pyproject.toml` menggunakan version ranges yang luas
- Beberapa dependencies tidak di-pin
- Potential compatibility issues

**Contoh dari `pyproject.toml`:**
```toml
dependencies = [
    "torch>=2.1.0,<2.2.0",      # Good - specific range
    "quran-transcript>=0.1.0",   # Bad - no upper bound
    "rich>=14.1.0",              # Bad - no upper bound
]
```

**Rekomendasi:**
```toml
dependencies = [
    "torch>=2.1.0,<2.3.0",
    "quran-transcript>=0.1.0,<0.2.0",
    "rich>=14.1.0,<15.0.0",
    "gradio>=5.43.1,<6.0.0",
]
```

Atau gunakan `uv.lock` (sudah ada!) secara konsisten:
```bash
uv pip compile pyproject.toml -o requirements.txt
uv pip sync requirements.txt
```

### 12. **Monkey Patching yang Berisiko**

**Masalah:**
- Monkey patch untuk `torchaudio.list_audio_backends` (line 23-26)
- Tidak ada dokumentasi mengapa ini diperlukan
- Bisa break di future versions

**Code:**
```python
# Monkey patch for recitations-segmenter compatibility with newer torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = list_audio_backends
```

**Rekomendasi:**
```python
# Lebih baik: Fork/fix the upstream library
# Atau: Document clearly dan create issue upstream
# Atau: Version pin torchaudio that works

# At minimum, add proper documentation
logger.warning(
    "Patching torchaudio.list_audio_backends for compatibility with "
    "recitations-segmenter. See issue #XXX. "
    "This patch may not be needed in torchaudio >= X.X.X"
)
```

---

## 🟢 Masalah Minor (Low Priority)

### 13. **Naming Conventions yang Tidak Konsisten**

**Masalah:**
- Mix antara English dan Indonesian untuk variable names
- Abbreviations yang tidak konsisten

**Contoh:**
```python
mv_sura_dropdown  # 'mv' = multi-verse? Tidak jelas
mv_start_aya      # Mix English 'aya' dengan Indonesia context
sura_idx_to_name  # Good
field_names       # Generic
```

**Rekomendasi:**
- Konsisten gunakan English untuk code
- Gunakan Indonesian untuk UI/user-facing strings
- Avoid abbreviations atau document clearly

### 14. **Commented Out Code**

**Masalah:**
- Banyak commented out code di tests (line 198-414 di `test_modules.py`)
- Commented out alternative launch di `main()` (line 1083, 1088)

**Lokasi:**
- `tests/test_modules.py` lines 198-414
- `gradio_app.py` lines 1083, 1088

**Rekomendasi:**
- Remove commented code atau move ke git history
- Jika masih useful, create proper TODO or feature flag

### 15. **Inline Styles di HTML Output**

**Masalah:**
- Banyak inline CSS di HTML output strings
- Hard to maintain consistency
- Tidak reusable

**Contoh:**
```python
html_output = f"<div style='border:1px solid #e5e7eb; margin-bottom:20px; padding:15px; border-radius:8px; background-color: white;'>"
```

**Rekomendasi:**
```python
# styles.py
CSS_CARD = "border:1px solid #e5e7eb; margin-bottom:20px; ..."
CSS_HEADER = "font-size: 28px; line-height: 1.7; ..."

# Atau better: gunakan CSS classes
CUSTOM_CSS = """
.verse-card { border: 1px solid #e5e7eb; ... }
.uthmani-text { font-size: 28px; ... }
"""
```

### 16. **TODO yang Tidak Diselesaikan**

**Masalah:**
- Ada TODO di `inference.py` line 131 yang tidak diselesaikan
- `# TODO: check input waves`

**Rekomendasi:**
- Implement validation atau remove TODO
- Jika nanti, create GitHub issue

```python
def validate_audio_input(waves: list) -> None:
    """Validate audio input format and data."""
    for idx, wave in enumerate(waves):
        if isinstance(wave, np.ndarray):
            if wave.ndim != 1:
                raise ValueError(f"Wave {idx} must be 1D, got {wave.ndim}D")
            if wave.size == 0:
                raise ValueError(f"Wave {idx} is empty")
        # ... more validations
```

### 17. **Logging yang Minimal**

**Masalah:**
- Only basic logging setup
- Tidak ada structured logging
- Tidak ada log levels yang proper
- Print statements dicampur dengan logging

**Contoh:**
```python
logging.basicConfig(level=logging.INFO)  # Too simple
print("Loading Recitation Segmenter...")  # Should use logger
```

**Rekomendasi:**
```python
import logging
import sys

def setup_logging(level=logging.INFO):
    """Setup structured logging with proper formatting."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('quran_muaalem.log')
        ]
    )

logger = logging.getLogger(__name__)

# Usage
logger.info("Loading Recitation Segmenter...")
logger.debug(f"Using device: {device}")
logger.warning("Segmentation mismatch: expected %d, got %d", expected, detected)
```

---

## ✅ Hal yang Sudah Baik

1. **Type Hints di Core Modules** - `inference.py` dan `decode.py` memiliki type hints yang baik
2. **Dataclasses Usage** - Penggunaan `@dataclass` untuk structured data
3. **Modular Model Architecture** - Separation of concerns di `modeling/` directory
4. **Comprehensive Tests untuk Core** - CTC decode dan inference memiliki tests yang solid
5. **Good README** - Dokumentasi instalasi dan usage yang jelas
6. **Proper Attribution** - Credit tab memberikan proper citation
7. **Use of uv.lock** - Modern dependency management dengan uv

---

## 🎯 Rekomendasi Prioritas

### Immediate (Minggu 1-2):
1. ✅ Add `build/` to `.gitignore`
2. ✅ Extract translations ke separate JSON files
3. ✅ Add basic CI/CD workflow
4. ✅ Fix global mutable state dengan Gradio State

### Short-term (Bulan 1):
1. ✅ Refactor `gradio_app.py` menjadi modules yang lebih kecil
2. ✅ Add integration tests untuk segmentation feature
3. ✅ Implement proper error handling dan logging
4. ✅ Create configuration management system

### Long-term (Bulan 2-3):
1. ✅ Improve test coverage ke >80%
2. ✅ Add performance monitoring
3. ✅ Consider containerization (Docker)
4. ✅ Setup proper deployment pipeline

---

## 📊 Metrics

```
Lines of Code (LOC):
├── gradio_app.py: 1089 baris (❌ TOO LARGE)
├── decode.py: 581 baris (⚠️ Could be split)
├── inference.py: 193 baris (✅ Good)
└── Total src/: ~1900 baris (excluding translations)

Test Coverage (estimated):
├── Core modules: ~60% (✅)
├── UI logic: ~0% (❌)
└── Overall: ~30% (❌ TARGET: 80%+)

Code Quality Issues:
├── Critical: 6
├── Medium: 6
└── Minor: 7
```

---

## 🏁 Kesimpulan

Repository ini memiliki **foundation yang solid** dari sisi machine learning dan core functionality. Modifikasi yang ditambahkan (Indonesian translation & multi-verse segmentation) adalah **value add yang bagus**. 

Namun, dari sisi **software engineering practices**, ada banyak ruang untuk improvement terutama di:
- **Code organization** (file yang terlalu besar)
- **State management** (global mutable state)
- **Testing** (coverage rendah untuk features baru)
- **Maintainability** (hardcoded translations, magic numbers)

Dengan refactoring yang proper, repository ini bisa menjadi production-ready dan lebih mudah untuk di-maintain serta dikembangkan lebih lanjut.

---

**Dibuat oleh:** AI Code Reviewer  
**Tanggal:** 29 November 2025
