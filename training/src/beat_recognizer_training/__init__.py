"""Training pipeline for the beat_recognizer WASM module's learned front-end.

See docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md and the build plan referenced
in that doc for the phased plan this package implements:

- data/   : license-clean audio acquisition (FMA, MTG-Jamendo CC subsets)
            and the provenance manifest that backs the "actually clean, not
            just asserted" licensing claim.
- label/  : label bootstrapping via beat-dsp's deterministic tracker
            (no NC-licensed "teacher" model involved, by design).
- model/  : the causal-TCN architecture and training loop.
- eval/   : mir_eval-based benchmarking against SMC/Osu2MIR/Ballroom/etc.
"""
