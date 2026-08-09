//! wasm-bindgen glue implementing the `BeatEstimate`/`BeatRecognizer` API
//! from `docs/WASM_BEAT_MODULE_SPEC.md` §4. Currently wraps `beat-dsp`'s
//! stub tracker only — Phase 1 fills in real DSP, Phase 7 adds `beat-model`'s
//! learned front-end on top.

use beat_dsp::DeterministicTracker;
use wasm_bindgen::prelude::*;

/// JS-facing mirror of `beat_dsp::BeatEstimate`, field names matching the
/// spec's TS interface exactly (camelCase).
#[wasm_bindgen]
pub struct BeatEstimate {
    inner: beat_dsp::BeatEstimate,
}

#[wasm_bindgen]
impl BeatEstimate {
    #[wasm_bindgen(getter)]
    pub fn bpm(&self) -> f32 {
        self.inner.bpm
    }

    #[wasm_bindgen(getter, js_name = beatPhase)]
    pub fn beat_phase(&self) -> f32 {
        self.inner.beat_phase
    }

    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    #[wasm_bindgen(getter, js_name = nextBeatTimeMs)]
    pub fn next_beat_time_ms(&self) -> f64 {
        self.inner.next_beat_time_ms
    }

    #[wasm_bindgen(getter, js_name = isDownbeat)]
    pub fn is_downbeat(&self) -> Option<bool> {
        self.inner.is_downbeat
    }
}

impl From<beat_dsp::BeatEstimate> for BeatEstimate {
    fn from(inner: beat_dsp::BeatEstimate) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
pub struct BeatRecognizer {
    tracker: DeterministicTracker,
}

#[wasm_bindgen]
impl BeatRecognizer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            tracker: DeterministicTracker::new(),
        }
    }

    #[wasm_bindgen(js_name = processAudioChunk)]
    pub fn process_audio_chunk(&mut self, samples: &[f32], sample_rate: u32) {
        self.tracker.process_chunk(samples, sample_rate);
    }

    #[wasm_bindgen(js_name = getCurrentEstimate)]
    pub fn get_current_estimate(&self) -> BeatEstimate {
        self.tracker.current_estimate().into()
    }

    pub fn reset(&mut self) {
        self.tracker.reset();
    }

    #[wasm_bindgen(js_name = analyzeOffline)]
    pub fn analyze_offline(&self, buffer: &[f32], sample_rate: u32) -> Vec<BeatEstimate> {
        self.tracker
            .analyze_offline(buffer, sample_rate)
            .into_iter()
            .map(BeatEstimate::from)
            .collect()
    }
}

impl Default for BeatRecognizer {
    fn default() -> Self {
        Self::new()
    }
}
