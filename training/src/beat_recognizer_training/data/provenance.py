"""Data provenance manifest: the audit trail behind the "every training input
is genuinely license-clean" claim (see docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md
§4). Every audio file that enters training must have a logged entry here
*before* use — the fetch scripts in this package call `append_entry` as part
of downloading, not as a manual follow-up step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Licenses that clear the bar in docs/WASM_BEAT_MODULE_SPEC.md §8: no NC, no
# ND, no "research use only". Anything not on this allowlist must not be
# written to the manifest by an automated fetch script.
ALLOWED_LICENSES = frozenset(
    {
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-3.0",
        "CC-BY-SA-4.0",
        "CC-BY-SA-3.0",
        "PUBLIC-DOMAIN",
        "OPEN-AUDIO-LICENSE",
        "FREE-ART-LICENSE",
    }
)


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
