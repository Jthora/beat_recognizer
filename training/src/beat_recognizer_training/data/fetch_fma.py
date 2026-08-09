"""Phase 2: pull the curated commercial-use subset of FMA (Free Music
Archive) — public-domain / CC-BY / CC-BY-SA / Open Audio License / Free Art
License tracks only. See docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4 for
why the full FMA dataset can't be used as-is (it mixes in NC/ND tracks).

TODO(Phase 2): implement against the curated commercial-use FMA subset,
writing a `data.provenance.append_entry()` call per downloaded file.
"""
