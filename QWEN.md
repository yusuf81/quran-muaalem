# Quran Muaalem Project Context

## Project Overview

Quran Muaalem is an AI-powered Quranic recitation analysis tool that helps detect pronunciation errors, tajweed rules, and phonetic characteristics in Quranic recitation. The project uses machine learning models to analyze phonetic features of Quranic verses and provide feedback on proper pronunciation.

### Key Features:
- Phonetic analysis of Quranic recitation
- Error detection for pronunciation, tajweed and phonetic characteristics
- Multi-level CTC (Connectionist Temporal Classification) architecture
- Compatible with the `quran-transcript` library for phonetic representation
- Gradio-based UI for interactive analysis

### Architecture:
- Multi-level CTC model architecture with 660M parameters
- Wav2Vec2BERT-based model for phonetic analysis
- Requires only 1.5GB of GPU memory
- Trained on Quranic phonetic script with recitation errors and tajweed rules

## Project Structure

```
/root/quran-muaalem/
├── src/
│   └── quran_muaalem/
│       ├── modeling/
│       ├── __init__.py
│       ├── decode.py
│       ├── explain_gradio.py
│       ├── explain.py
│       ├── gradio_app.py
│       ├── inference.py
│       └── muaalem_typing.py
├── tests/
├── pyproject.toml
├── README.md
├── uv.lock
└── assets/
    └── test.wav
```

### Key Source Files:
- `inference.py`: Main Muaalem class for model inference
- `gradio_app.py`: Gradio interface for the web application
- `explain.py`: Functions for explaining recitation analysis
- `modeling/`: Model architecture and tokenizer implementations
- `muaalem_typing.py`: Type definitions for outputs

## Building and Running

### Prerequisites:
- Python 3.10+
- System dependencies: `ffmpeg`, `libsndfile1`, `portaudio19-dev`
- PyTorch (compatible with CUDA if available)

### Installation:
```bash
# Install system dependencies
sudo apt-get install -y ffmpeg libsndfile1 portaudio19-dev

# Install Python packages
pip install quran-muaalem librosa "numba>=0.61.2"
```

### Running with Gradio UI:
```bash
# Using uvx (recommended)
uvx --no-cache --from https://github.com/obadx/quran-muaalem.git[ui] quran-muaalem-ui
```

### Python API Usage:
```python
from quran_muaalem import Muaalem
from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from librosa.core import load

# Initialize the model
muaalem = Muaalem(device="cuda" if torch.cuda.is_available() else "cpu")

# Load audio
wave, _ = load("./assets/test.wav", sr=16000, mono=True)

# Prepare reference text
uthmani_ref = Aya(8, 75).get_by_imlaey_words(17, 9).uthmani
moshaf = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=2,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=2,
)
phonetizer_out = quran_phonetizer(uthmani_ref, moshaf, remove_spaces=True)

# Analyze recitation
outs = muaalem([wave], [phonetizer_out], sampling_rate=16000)
```

## Development Conventions

- The project uses Hugging Face Transformers for the underlying model architecture
- Audio sampling rate must be 16000Hz
- Phonetic analysis is based on the `quran-transcript` library
- Multi-level CTC model handles different phonetic features simultaneously
- Gradio UI allows for interactive analysis with Arabic language support
- Tests are located in the `tests/` directory and can be run with pytest

## Testing

Run tests using pytest:
```bash
pip install pytest
pytest tests/
```