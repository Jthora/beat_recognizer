# data/

Local cache for training audio, populated by `training/src/beat_recognizer_training/data/fetch_fma.py`
and `fetch_jamendo.py` (Phase 2 of the build plan). The audio itself is never
committed — it's regenerable from `PROVENANCE.jsonl` plus the fetch scripts.

## PROVENANCE.jsonl

The audit trail behind this project's licensing claim: **every** file that
enters training must have a line here before use, written automatically by
the fetch scripts via `beat_recognizer_training.data.provenance.append_entry`,
never added by hand. Each line is a JSON object:

```json
{"file_path": "...", "source_url": "...", "license": "CC-BY-4.0", "fetched_at": "2026-08-09T18:00:00+00:00", "sha256": "...", "source_dataset": "fma"}
```

`license` must be one of the values in `provenance.ALLOWED_LICENSES` — the
commercial-use allowlist from `docs/WASM_BEAT_MODULE_SPEC.md` §8 (no NC, no
ND, no research-only). `append_entry` raises if it isn't; nothing should be
able to reach this file without clearing that bar first.

This file *is* committed (it's small, text-only, and is the whole point —
see `docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md` §4/§9).
