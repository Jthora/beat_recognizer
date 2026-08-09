"""Phase 2: pull MTG-Jamendo tracks, filtered per-track to CC-BY / CC0 /
CC-BY-SA only. MTG-Jamendo mixes CC license variants including NC and ND —
the filter is mandatory, not a formality (see
docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4).

TODO(Phase 2): implement against the MTG-Jamendo metadata API/dump, checking
each track's specific license field before download, writing a
`data.provenance.append_entry()` call per downloaded file.
"""
