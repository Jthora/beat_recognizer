# giantsteps -- offline vs. streaming (production readiness item 2)

664 tracks. Offline: 664 scored. Streaming: 664 scored.

| metric | offline | streaming | gap |
|---|---|---|---|
| Accuracy1 | 52.3% | 39.0% | -13.3% |
| Accuracy2 | 82.1% | 71.4% | -10.7% |
| Accuracy2+alt | 86.4% | 75.0% | -11.4% |

Streaming numbers reflect the *final* estimate after processing the whole track in 512-sample chunks through the causal tracker -- a lower bound on real-time behavior, since production also cares about how fast that estimate converges after track start, which this doesn't measure (see the build plan's production-readiness item 2 for the time-to-lock follow-up).
