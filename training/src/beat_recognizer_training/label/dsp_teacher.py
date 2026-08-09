"""Phase 3: bootstrap beat/downbeat labels for the Phase 2 audio corpus using
beat-dsp's deterministic tracker — deliberately not an NC-licensed ML model,
so there's no licensing gray area to manage (see
docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4 and the approved build plan's
Phase 3). Confidence-filters low-salience clips and expects a human
spot-check pass on a sample before the output is trusted for training.

Phase 5 repeats this module's role using the Phase 4 trained model instead
(now fully in-house, so no licensing question either way).

TODO(Phase 3): shell out to (or FFI-bind) beat-dsp's offline analyzer,
running it across the Phase 2 corpus and writing confidence-filtered
(audio_path, beat_times, downbeat_times, confidence) records for training/model.
"""
