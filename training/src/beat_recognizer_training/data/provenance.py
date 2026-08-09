"""Data provenance manifest: the audit trail behind the "every training input
is genuinely license-clean" claim (see docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md
§4). Every audio file that enters training must have a logged entry here
*before* use — the fetch scripts in this package call `append_entry` as part
of downloading, not as a manual follow-up step.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Licenses that clear the bar in docs/WASM_BEAT_MODULE_SPEC.md §8: no NC, no
# ND, no "research use only". Anything not on this allowlist must not be
# written to the manifest by an automated fetch script.
#
# Every published Creative Commons Attribution / Attribution-ShareAlike
# version (1.0 through 4.0) is included, not just the more common 3.0/4.0 —
# CC licenses are irrevocable once applied, so a track licensed under CC BY
# 2.0 is exactly as commercially usable today as one under CC BY 4.0. An
# earlier version of this allowlist only had 3.0/4.0, which turned out to
# reject real, perfectly valid CC BY 2.0/2.5 tracks in FMA's metadata for no
# actual legal reason — caught by running the real fetch script against real
# data, not assumed correct from the start.
_CC_VERSIONS = ("1.0", "2.0", "2.5", "3.0", "4.0")
ALLOWED_LICENSES = frozenset(
    {"CC0-1.0", "PUBLIC-DOMAIN", "OPEN-AUDIO-LICENSE", "FREE-ART-LICENSE"}
    | {f"CC-BY-{v}" for v in _CC_VERSIONS}
    | {f"CC-BY-SA-{v}" for v in _CC_VERSIONS}
)

_CC_URL_RE = re.compile(
    r"creativecommons\.org/(?:licenses/([a-z-]+)|publicdomain/zero)/(\d\.\d)"
)


def classify_cc_url(url: str | None) -> str | None:
    """Parses a Creative Commons license URL (as returned by FMA's
    license_url or Jamendo's license_ccurl fields) into one of
    ALLOWED_LICENSES, or None if it's NC/ND/unrecognized. The single source
    of truth for CC-URL classification, shared by every fetch script so the
    same license logic isn't duplicated — and potentially drifting — per
    data source."""
    if not url:
        return None
    m = _CC_URL_RE.search(url.lower())
    if not m:
        return None
    kind, version = m.groups()
    if kind is None:  # publicdomain/zero URL shape
        return "CC0-1.0"
    if kind == "by":
        return f"CC-BY-{version}"
    if kind == "by-sa":
        return f"CC-BY-SA-{version}"
    return None  # by-nc*, by-nd*, by-nc-sa*, etc.


@dataclass(frozen=True)
class ProvenanceEntry:
    file_path: str
    source_url: str
    license: str
    fetched_at: str
    sha256: str
    source_dataset: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_entry(
    manifest_path: Path,
    *,
    file_path: Path,
    source_url: str,
    license: str,
    source_dataset: str,
) -> ProvenanceEntry:
    if license not in ALLOWED_LICENSES:
        raise ValueError(
            f"license {license!r} for {file_path} is not on the commercial-use "
            f"allowlist ({sorted(ALLOWED_LICENSES)}) — refusing to log it as "
            "training data. See spec §8."
        )

    entry = ProvenanceEntry(
        file_path=str(file_path),
        source_url=source_url,
        license=license,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        sha256=sha256_of(file_path),
        source_dataset=source_dataset,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")

    return entry
