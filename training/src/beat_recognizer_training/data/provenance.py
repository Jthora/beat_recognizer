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
# CC-BY-*/CC-BY-SA-* are matched by *any* version number (see is_allowed),
# not a fixed enumerated list — jurisdiction "ports" of a given CC license
# (e.g. a country-specific legal translation) sometimes carry unusual
# version numbers like 2.1 or 2.7 that have nothing to do with permission
# scope: that's fixed entirely by the by/nc/nd/sa combination in the URL,
# already resolved by the time a version number is being read at all. A
# real Australian "CC-BY-SA-2.1" track crashed an earlier version of this
# check that only recognized 1.0/2.0/2.5/3.0/4.0 — confirmed by hitting it
# in an actual data pull, not hypothesized.
ALLOWED_LICENSES = frozenset(
    {"CC0-1.0", "PUBLIC-DOMAIN", "OPEN-AUDIO-LICENSE", "FREE-ART-LICENSE"}
    | {f"CC-BY-{v}" for v in ("1.0", "2.0", "2.5", "3.0", "4.0")}
    | {f"CC-BY-SA-{v}" for v in ("1.0", "2.0", "2.5", "3.0", "4.0")}
)

_VERSIONED_CC_RE = re.compile(r"^CC-BY(-SA)?-\d(\.\d+)?$")


def is_allowed(license: str) -> bool:
    """The real gate `append_entry` enforces: exact membership in
    ALLOWED_LICENSES for the fixed non-versioned categories, but *any*
    version number for CC-BY/CC-BY-SA, per the reasoning above."""
    return license in ALLOWED_LICENSES or bool(_VERSIONED_CC_RE.match(license))

_CC_URL_RE = re.compile(
    r"creativecommons\.org/(?:licenses/([a-z-]+)|publicdomain/zero)/(\d\.\d)"
)


def classify_cc_url(url: str | None) -> str | None:
    """Parses a license URL (as returned by FMA's license_url or Jamendo's
    license_ccurl fields) into one of ALLOWED_LICENSES, or None if it's
    NC/ND/unrecognized. The single source of truth for license-URL
    classification, shared by every fetch script so the same logic isn't
    duplicated — and potentially drifting — per data source."""
    if not url:
        return None
    url_lower = url.lower()

    if "artlibre.org" in url_lower:
        # Free Art License (Licence Art Libre) — an explicitly
        # commercial-use-permitting copyleft license, distinct from CC but
        # already on ALLOWED_LICENSES; caught missing here after seeing 33
        # real Jamendo tracks under it silently rejected by omission rather
        # than a real evaluation of the license terms.
        return "FREE-ART-LICENSE"

    m = _CC_URL_RE.search(url_lower)
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


def load_logged_paths(manifest_path: Path) -> set[str]:
    """Every file_path already present in the manifest, so a fetch script
    interrupted partway through (network error, one bad track in a batch of
    thousands) can resume without re-downloading or double-logging files it
    already finished — check membership once up front rather than
    re-scanning the manifest per file."""
    if not manifest_path.exists():
        return set()
    paths = set()
    with manifest_path.open() as f:
        for line in f:
            if line.strip():
                paths.add(json.loads(line)["file_path"])
    return paths


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
    if not is_allowed(license):
        raise ValueError(
            f"license {license!r} for {file_path} is not on the commercial-use "
            f"allowlist ({sorted(ALLOWED_LICENSES)}, or any CC-BY(-SA)-<version>) "
            "— refusing to log it as training data. See spec §8."
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
