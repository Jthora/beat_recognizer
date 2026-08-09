# training/

Python training pipeline for the beat_recognizer WASM module's learned
front-end (Phases 2-6 of the build plan referenced in
`docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md`). Managed with `uv`.

```
uv sync   # installs torch, torchaudio, mir_eval, numpy, requests into .venv
```

See `src/beat_recognizer_training/__init__.py` for the module layout.
