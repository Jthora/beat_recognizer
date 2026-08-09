Beat Recognizer
================

A real-time and offline music beat/tempo recognizer, compiled to a
TypeScript-callable WASM module, built for [ProtoFusionGirl](docs/FUSIONGIRL_MUSIC_VISION.md).
License-clean from the ground up — see `docs/` for why that's the whole
point of this project.

## Layout

- `docs/` — requirements spec, architecture/licensing research, and the
  approved build plan.
- `crates/` — Rust workspace: `beat-dsp` (deterministic DSP tracking, no ML),
  `beat-model` (learned front-end, loaded via `tract`), `beat-wasm`
  (wasm-bindgen glue implementing the public API).
- `ts/` — the TypeScript-callable package (`pnpm build`).
- `training/` — Python pipeline for data acquisition, label bootstrapping,
  model training, and evaluation (`uv sync`).
- `data/` — local, gitignored audio cache + the license provenance manifest.

## Start here

`docs/WASM_BEAT_MODULE_SPEC.md` for what this builds and why;
`docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md` for the architecture and
data-licensing decisions behind how it's being built.
