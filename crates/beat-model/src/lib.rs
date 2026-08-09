//! Learned front-end: loads the exported causal-TCN weights (trained in
//! `training/`, see docs/ build plan Phases 2-5) and produces a beat/downbeat
//! activation curve for `beat-dsp`'s deterministic tracking layer to consume.
//!
//! Empty until Phase 7 — no trained model exists yet. `beat-wasm` runs on
//! `beat-dsp` alone until then.

pub use beat_dsp::BeatEstimate;
