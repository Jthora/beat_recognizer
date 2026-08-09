// Thin typed wrapper over the wasm-bindgen output, matching the API shape
// from docs/WASM_BEAT_MODULE_SPEC.md §4. Phase 0 stub: wraps beat-dsp's
// placeholder tracker only (always reports confidence 0 — no ML front-end,
// no real DSP yet; see crates/beat-dsp for the Phase 1 TODOs).

import init, {
  BeatRecognizer as WasmBeatRecognizer,
  BeatEstimate as WasmBeatEstimate,
} from "../wasm/beat_wasm.js";

export interface BeatEstimate {
  bpm: number;
  beatPhase: number;
  confidence: number;
  nextBeatTimeMs: number;
  isDownbeat?: boolean;
}

function toBeatEstimate(estimate: WasmBeatEstimate): BeatEstimate {
  return {
    bpm: estimate.bpm,
    beatPhase: estimate.beatPhase,
    confidence: estimate.confidence,
    nextBeatTimeMs: estimate.nextBeatTimeMs,
    isDownbeat: estimate.isDownbeat ?? undefined,
  };
}

/**
 * Real-time and offline beat/tempo recognizer. WASM instantiation is async,
 * so construct via `BeatRecognizer.create()` rather than `new`.
 */
export class BeatRecognizer {
  private constructor(private readonly inner: WasmBeatRecognizer) {}

  static async create(): Promise<BeatRecognizer> {
    await init();
    return new BeatRecognizer(new WasmBeatRecognizer());
  }

  /** Streaming mode: feed successive chunks of audio as they arrive. */
  processAudioChunk(samples: Float32Array, sampleRate: number): void {
    this.inner.processAudioChunk(samples, sampleRate);
  }

  getCurrentEstimate(): BeatEstimate {
    return toBeatEstimate(this.inner.getCurrentEstimate());
  }

  /** Call when switching tracks/sources. */
  reset(): void {
    this.inner.reset();
  }

  /** Offline/batch mode: analyze a complete pre-loaded buffer in one pass. */
  analyzeOffline(buffer: Float32Array, sampleRate: number): BeatEstimate[] {
    return Array.from(this.inner.analyzeOffline(buffer, sampleRate)).map(
      toBeatEstimate,
    );
  }
}
