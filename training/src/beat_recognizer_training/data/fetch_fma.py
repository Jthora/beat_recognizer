"""Phase 2: pull the commercial-use-clean subset of FMA (Free Music Archive).

FMA distributes metadata and audio as bulk archives, not a per-track API —
see https://github.com/mdeff/fma. This module:

1. Downloads `fma_metadata.zip` (~342MB) and reads `tracks.csv`, which has a
   `(track, license)` free-text field per track (e.g. "Attribution",
   "Attribution-NonCommercial-ShareAlike") and, when present, a
   `(track, license_url)` pointing at the actual Creative Commons URL.
2. Classifies each track's license, deliberately failing *closed*: a track
   is only accepted if its license can be positively identified as clean
   per `provenance.ALLOWED_LICENSES` (spec §8) — NC/ND markers are rejected
   outright, and anything ambiguous (e.g. "Attribution" with no confirmable
   CC version, since the plain-text field alone doesn't say 3.0 vs 4.0) is
   rejected for manual review rather than guessed at. This is slower than
   trusting the plain-text field, but the whole point of this project is
   not repeating the "commonly used, never actually verified" mistake
   documented in docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4.
3. Only once a track list is confirmed clean does it extract the
   corresponding audio from one of FMA's size-tiered archives
   (fma_small/medium/large/full — see AUDIO_TIER_URLS), logging each
   extracted file via `provenance.append_entry`.

The metadata step is small and safe to run freely. The audio-archive step
downloads multiple gigabytes — this module never triggers that
automatically; it's a separate, explicit call (see `download_audio_tier`),
meant to be run only after a human has reviewed the license report from
step 2 (`report_license_distribution`).
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from . import provenance

FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

# Larger tiers are supersets by track count, not by content — fma_small's
# tracks are a different (smaller, curated) selection than fma_large's, so
# picking a tier is a real choice, not just "bigger is a superset of smaller".
AUDIO_TIER_URLS = {
    "small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",  # 7.2GB, 8000 tracks, 30s clips
    "medium": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",  # 22GB, 25000 tracks, 30s clips
    "large": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",  # 93GB, 106574 tracks, 30s clips
    "full": "https://os.unil.cloud.switch.ch/fma/fma_full.zip",  # 879GB, full-length tracks
}

# Plain-text `license` field markers that unambiguously disqualify a track
# regardless of anything else — checked first, fail-closed.
_DENY_MARKERS = (
    "noncommercial",
    "non-commercial",
    "noderiv",
    "no-deriv",
    "no derivative",  # e.g. FMA's "Attribution-No Derivative Works 3.0" —
    # caught by re-checking real classifier output line by line, not just
    # trusting the aggregate count: "noderiv" alone missed this
    # space-separated phrasing and wrongly accepted 59 ND-licensed tracks.
    "sampling",
    "nc-sa",
    "nc-nd",
    "nc)",
)


@dataclass(frozen=True)
class FmaTrack:
    track_id: str
    raw_license: str
    license_url: str | None


def download_metadata(dest_dir: Path) -> Path:
    """Downloads and extracts fma_metadata.zip, returning the path to
    tracks.csv. Small (~342MB) — safe to run without asking."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "fma_metadata.zip"
    if not zip_path.exists():
        _download(FMA_METADATA_URL, zip_path)

    extract_dir = dest_dir / "fma_metadata"
    tracks_csv = extract_dir / "tracks.csv"
    if not tracks_csv.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    return tracks_csv


def load_tracks(tracks_csv: Path) -> list[FmaTrack]:
    """Parses tracks.csv's 3-row multi-index header to pull out track id,
    license, and license_url (when present) without needing pandas."""
    with tracks_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Row 0: category (e.g. "track"), row 1: field name (e.g. "license"),
    # row 2: sub-type — the actual header FMA uses for this file.
    category_row, field_row = rows[0], rows[1]
    header = list(zip(category_row, field_row))

    def col_index(category: str, field: str) -> int | None:
        for i, (c, f) in enumerate(header):
            if c == category and f == field:
                return i
        return None

    license_idx = col_index("track", "license")
    license_url_idx = col_index("track", "license_url")
    if license_idx is None:
        raise RuntimeError(
            "tracks.csv doesn't have the expected (track, license) column — "
            "FMA's metadata schema may have changed; verify before proceeding."
        )

    tracks = []
    for row in rows[3:]:  # first 3 rows are header levels
        if not row or not row[0]:
            continue
        track_id = row[0]
        raw_license = row[license_idx] if license_idx < len(row) else ""
        license_url = (
            row[license_url_idx]
            if license_url_idx is not None and license_url_idx < len(row)
            else None
        )
        tracks.append(FmaTrack(track_id=track_id, raw_license=raw_license, license_url=license_url or None))
    return tracks


_VERSION_RE = re.compile(r"(\d\.\d)")


def classify_license(track: FmaTrack) -> str | None:
    """Returns a value from provenance.ALLOWED_LICENSES, or None if the
    track should be rejected (unclear, NC, ND, or otherwise not confirmed
    clean).

    Checked `track.license_url` first when present (an explicit CC URL is
    the most authoritative signal), but FMA's actually-distributed
    tracks.csv has no such column — confirmed by inspecting the real header
    row, not assumed — so in practice this always falls through to parsing
    the plain-text `license` field directly. FMA's own naming convention
    spells the CC version out in that text (e.g. "Attribution 3.0
    International", "Attribution-ShareAlike 2.5 Canada"), so it's a real
    source of truth, not a guess — as long as NC/ND are excluded first and
    the version number is actually present and parseable."""
    raw = (track.raw_license or "").strip()
    raw_lower = raw.lower()

    if not raw:
        return None
    if any(marker in raw_lower for marker in _DENY_MARKERS):
        return None

    if track.license_url:
        classified = provenance.classify_cc_url(track.license_url)
        if classified:
            return classified

    if "cc0" in raw_lower or "public domain" in raw_lower or "commons 0" in raw_lower:
        # "Public Domain / Sound Recording Common Law Protection" is
        # deliberately excluded from this match despite containing "public
        # domain": some jurisdictions (notably the US, pre-1972 recordings)
        # reserve separate common-law rights over the recording itself even
        # when the underlying composition is public domain — not something
        # to wave through on a substring match.
        if "common law" in raw_lower:
            return None
        return "CC0-1.0"

    if "attribution" in raw_lower:
        version_match = _VERSION_RE.search(raw)
        if not version_match:
            return None  # can't confirm a version — reject, don't guess
        version = version_match.group(1)
        is_sharealike = "share alike" in raw_lower or "sharealike" in raw_lower
        tag = f"CC-BY-SA-{version}" if is_sharealike else f"CC-BY-{version}"
        return tag if provenance.is_allowed(tag) else None

    return None


def report_license_distribution(tracks: list[FmaTrack]) -> None:
    """Prints every distinct raw license string encountered and how it was
    classified, so a human reviews the classifier's actual behavior against
    real data rather than trusting it blindly."""
    from collections import Counter

    counts: Counter[str] = Counter()
    outcomes: dict[str, str] = {}
    for t in tracks:
        counts[t.raw_license] += 1
        result = classify_license(t)
        outcomes[t.raw_license] = result if result else "REJECTED"

    print(f"{'license string':<55} {'count':>7}  outcome")
    print("-" * 80)
    for raw, count in counts.most_common():
        print(f"{raw[:55]:<55} {count:>7}  {outcomes[raw]}")

    accepted = sum(1 for t in tracks if classify_license(t))
    print(f"\n{accepted} / {len(tracks)} tracks classified as commercial-use-clean.")


def _fma_relative_path(track_id: str) -> str:
    padded = track_id.zfill(6)
    return f"{padded[:3]}/{padded}.mp3"


def download_audio_tier(
    tier: str,
    clean_tracks: list[FmaTrack],
    dest_dir: Path,
    manifest_path: Path,
) -> None:
    """Downloads the given FMA audio archive tier and extracts *only* the
    tracks in `clean_tracks`, logging each to the provenance manifest. This
    is the multi-gigabyte step — call it only after reviewing
    `report_license_distribution`'s output."""
    if tier not in AUDIO_TIER_URLS:
        raise ValueError(f"unknown tier {tier!r}, expected one of {list(AUDIO_TIER_URLS)}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"fma_{tier}.zip"
    if not zip_path.exists():
        _download(AUDIO_TIER_URLS[tier], zip_path)

    clean_by_suffix = {_fma_relative_path(t.track_id): t for t in clean_tracks}
    audio_dir = dest_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    already_logged = provenance.load_logged_paths(manifest_path)

    extracted = 0
    skipped_existing = 0
    failed = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            for suffix, track in clean_by_suffix.items():
                if member.endswith(suffix):
                    out_path = audio_dir / suffix.replace("/", "_")
                    if str(out_path) in already_logged:
                        skipped_existing += 1
                        break
                    try:
                        if not out_path.exists():
                            with zf.open(member) as src, out_path.open("wb") as dst:
                                dst.write(src.read())
                        provenance.append_entry(
                            manifest_path,
                            file_path=out_path,
                            source_url=f"{AUDIO_TIER_URLS[tier]}#{member}",
                            license=classify_license(track),  # type: ignore[arg-type]
                            source_dataset=f"fma_{tier}",
                        )
                        extracted += 1
                    except (OSError, zipfile.BadZipFile, ValueError) as e:
                        print(f"  skipping {member} after error: {e}")
                        failed += 1
                    break

    print(
        f"Extracted {extracted} clean tracks from fma_{tier}.zip into {audio_dir} "
        f"({skipped_existing} already done, {failed} failed and were skipped)"
    )


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:.0f}MB / {total / 1e6:.0f}MB", end="", flush=True)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/fma"))
    parser.add_argument(
        "--download-tier",
        choices=list(AUDIO_TIER_URLS),
        default=None,
        help="If set, also downloads and extracts this audio tier (multi-GB). "
        "Omit to only fetch metadata and print the license report.",
    )
    args = parser.parse_args()

    tracks_csv = download_metadata(args.dest)
    tracks = load_tracks(tracks_csv)
    report_license_distribution(tracks)

    if args.download_tier:
        clean = [t for t in tracks if classify_license(t)]
        download_audio_tier(
            args.download_tier,
            clean,
            args.dest,
            Path("data/PROVENANCE.jsonl"),
        )
