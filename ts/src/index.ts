// Thin typed wrapper over the wasm-bindgen output, matching the API shape
// from docs/WASM_BEAT_MODULE_SPEC.md §4, extended per
// docs/WASM_BEAT_MODULE_ENHANCEMENTS.md §2 with hasRhythm and the
// alternate-octave fields (both additive — nothing here changes the
// original v1 field meanings). Wraps beat-dsp's deterministic tracker
// (Phase 1); the learned front-end lands in Phase 7.

import init, {
  BeatRecognizer as WasmBeatRecognizer,
  BeatEstimate as WasmBeatEstimate,
} from "../wasm/beat_wasm.js";

export interface BeatEstimate {
  /**
   * Whether any periodicity was detected at all. `false` means "no
   * discernible rhythm" (silence, talk radio, pure noise, or simply not
   * enough history yet) — distinct from a detected-but-low
   * `confidence`, which alone can't disambiguate those two cases. When
   * `false`, every other field below is meaningless and should be ignored
   * rather than treated as a low-quality reading.
   */
  hasRhythm: boolean;
  bpm: number;
  beatPhase: number;
  confidence: number;
  /**
   * Milliseconds since `BeatRecognizer.create()` or the last `reset()` —
   * *not* wall-clock time and not `AudioContext.currentTime` directly.
   * Add your own clock offset (e.g. the `AudioContext.currentTime` at the
   * moment you called `reset()`) to translate into your audio clock's
   * domain. Per docs/WASM_BEAT_MODULE_ENHANCEMENTS.md §5, this translation
   * is the caller's responsibility, not implicit.
   */
  nextBeatTimeMs: number;
  isDownbeat?: boolean;
  /**
   * The other tempo-octave candidate (e.g. 76 vs 152 BPM), present only
   * when the primary estimate is genuinely ambiguous — a real, common
   * tempo-perception phenomenon, not an estimation failure. Pick whichever
   * octave fits your context (e.g. a fast flight loop vs. a walk cycle
   * might want opposite octaves off the same audio).
   */
  alternateBpm?: number;
  alternateConfidence?: number;
}

function toBeatEstimate(estimate: WasmBeatEstimate): BeatEstimate {
  return {
    hasRhythm: estimate.hasRhythm,
    bpm: estimate.bpm,
    beatPhase: estimate.beatPhase,
    confidence: estimate.confidence,
    nextBeatTimeMs: estimate.nextBeatTimeMs,
    isDownbeat: estimate.isDownbeat ?? undefined,
    alternateBpm: estimate.alternateBpm ?? undefined,
    alternateConfidence: estimate.alternateConfidence ?? undefined,
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
