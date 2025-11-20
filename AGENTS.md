# Repository Guidelines

This guide keeps contributors aligned on structure, tooling, and review expectations for Quran Muaalem.

## Project Structure & Modules
- Core package lives in `src/quran_muaalem/` with decoding (`decode.py`), inference entrypoints (`inference.py`), explainability (`explain*.py`), and model pieces under `modeling/` (configuration, tokenizer, vocab, multilevel CTC model).
- Gradio UI launcher is `src/quran_muaalem/gradio_app.py` (also exposed via the `quran-muaalem-ui` console script).
- Tests sit in `tests/` as `test_*.py` plus shared fixtures in `conftest.py`.
- Visual assets are in `assets/`, and generated build artifacts (if any) land under `build/`.

## Setup, Build & Local Runs
- Python 3.10–3.13 is supported. System deps for audio: `ffmpeg libsndfile1 portaudio19-dev`.
- Typical editable install with tests:  
  ```bash
  uv venv && source .venv/bin/activate
  uv pip install -e .[test]
  ```
- Run Gradio UI locally (pulls model from Hugging Face):  
  ```bash
  uvx quran-muaalem[ui] quran-muaalem-ui
  ```
- Package install without UI extras: `python -m pip install .`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and type hints; mirror existing docstring style in `inference.py` and `modeling/*`.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and prefer descriptive logging via the standard `logging` module.
- Keep tensors and tokenizer interactions consistent with existing helpers (e.g., `MultiLevelTokenizer`, `Wav2Vec2BertForMultilevelCTC`); avoid duplicating vocab handling.

## Testing Guidelines
- Tests use `pytest`; target fast, deterministic cases around phoneme alignment, decoding, and sifat handling.
- Naming: `test_<thing>.py` with function-level tests matching the public API surface.
- Run the suite before every PR (system deps installed):  
  ```bash
  pytest -v
  ```
- Provide focused fixtures instead of large audio files; reuse helpers from `conftest.py` when extending coverage.

## Commit & Pull Request Guidelines
- Commit messages follow conventional prefixes (`feat:`, `fix:`, `chore:`, etc.) as seen in `git log`.
- For PRs: include a short summary of changes, link related issues, and paste the exact commands/tests you ran. Add screenshots or terminal output for UI or explainability changes.
- Avoid committing large binaries or model checkpoints—reference hosted models like `obadx/muaalem-model-v3_2` instead.
