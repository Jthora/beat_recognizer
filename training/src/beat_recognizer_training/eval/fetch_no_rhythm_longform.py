"""Production-readiness item 1/2, Iteration 3 item 3: `duration_sensitivity.py`
tried to test whether the no-rhythm false-positive rate drops with more
available audio by concatenating same-category ESC-50 clips -- but that just
tested "does a splice cause false detections" (a real, different, also
interesting question), since ESC-50 only ships 5s clips and has no
multi-minute continuous recordings to test the original question with.

This pulls genuinely continuous, long-duration, non-periodic field
recordings from Wikimedia Commons instead. Two properties make it a good
fit, checked directly rather than assumed:

- Every file on Commons is required by platform-wide policy to carry a real
  free license (no NC/ND mixed in the way Freesound's general pool has) --
  confirmed per-file via the same `provenance.classify_cc_url` used for
  every other source in this project, not a new, separately-trusted rule.
- The API's `imageinfo` endpoint exposes real per-file `duration` directly
  (confirmed live: results here range from 10s to 1800s/30min), letting
  clips be selected specifically to *not* need splicing.

Candidates were hand-picked (not scraped in bulk) from Commons' "Sounds of
nature" / "Ambient sounds" categories plus a few targeted searches, and
deliberately exclude anything with a plausible internal periodicity --
crickets, regular birdsong, sirens -- the same discipline already used to
build fetch_no_rhythm_set.py's ESC-50 category allowlist. This is a small,
manually-vetted validation set, not a bulk corpus source -- doesn't need
PROVENANCE.jsonl treatment the way training-input sources do (eval-only),
but is still license-filtered because it's cheap and keeps the discipline
consistent with the ESC-50 set.
"""

from __future__ import annotations

import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from beat_recognizer_training.data import provenance

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "beat_recognizer research/1.0 (jordan.trana@gmail.com)"

# Hand-picked, license-clean, non-periodic-by-nature. Excludes crickets,
# regular birdsong, and sirens found during the same search pass -- same
# exclusion discipline as ESC50_SAFE_CATEGORIES.
CANDIDATE_TITLES = [
    "File:Atmo – Ufer Wellen.mp3",
    "File:Cafe ambiance.ogg",
    "File:Atmo – Gewächshaus.mp3",
    "File:Atmo – Gewächshaus, Raum.mp3",
    "File:2013-07-24 00-19-04hrs Chiang Mai Chang Khien rain animal sounds night time.ogg",
    "File:2023-Mai-20 14h Wien innerer Lerchenfelder Gürtel Thaliagasse rumpeln der Straßenbahn 46.wav",
    "File:2023-Mai-20 14h34min Wien innerer Lerchenfelder Gürtel Thaliagasse Nachmittagsverkehr.wav",
    "File:2023-Mai-20 15h35min Stephansplatz voller Gemurmel.wav",
    "File:2023-Mai-20 15h40min Wien im linken Seitenschiff des Stephansdoms, fast Stille.wav",
    "File:Urban Street on a Rainy Afternoon.flac",
    "File:Ambience noise at 1 am in Bengaluru, Karnataka, India.wav",
    "File:20090610 0 ambience.ogg",
    "File:Forest ambience (Gravity Sound).wav",
    "File:Forest ambience 2 (Gravity Sound).wav",
    "File:Forest ambience 3 (Gravity Sound).wav",
]

MAX_CLIP_S = 90.0  # cap long files (one is 30min) -- plenty past the 5s/15s ESC-50 comparison point
SAMPLE_RATE = 44_100


@dataclass
class CommonsFile:
    title: str
    duration_s: float
    license: str
    url: str


def fetch_metadata(titles: list[str]) -> list[CommonsFile]:
    out = []
    for title in titles:
        params = {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mediatype",
            "format": "json",
        }
        resp = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            ext = info.get("extmetadata", {})
            license_url = ext.get("LicenseUrl", {}).get("value")
            license_short = ext.get("LicenseShortName", {}).get("value", "")
            classified = provenance.classify_cc_url(license_url)
            # Commons' own "Public domain" tag (no CC URL attached) is a
            # platform-enforced status distinct from an unclear license --
            # accepted directly, same bar as CC0.
            if classified is None and license_short.strip().lower() != "public domain":
                print(f"  skip (license): {title} ({license_short!r})")
                continue
            duration = info.get("duration")
            url = info.get("url")
            if not duration or not url:
                print(f"  skip (no duration/url): {title}")
                continue
            out.append(CommonsFile(title=page.get("title", title), duration_s=float(duration), license=classified or "PD", url=url))
    return out


def download_and_convert(cf: CommonsFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in cf.title.removeprefix("File:"))
    raw_path = dest_dir / f"_raw_{safe_name}"
    wav_path = dest_dir / (Path(safe_name).stem + ".wav")
    if wav_path.exists():
        return wav_path

    resp = requests.get(cf.url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    raw_path.write_bytes(resp.content)

    duration_arg = str(min(cf.duration_s, MAX_CLIP_S))
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(raw_path),
            "-t", duration_arg,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav",
            str(wav_path),
        ],
        check=True,
    )
    raw_path.unlink()
    return wav_path


def build_short_variants(long_paths: list[Path], dest_dir: Path, short_s: float = 5.0) -> list[Path]:
    """A 5s-cut prefix of each long clip -- the same source content as the
    long version, so any FP-rate difference between the two is attributable
    to duration alone, not to using a different underlying recording."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in long_paths:
        samples, sr = sf.read(str(p), dtype="float32")
        cut = samples[: int(short_s * sr)]
        out_path = dest_dir / p.name
        pcm16 = (np.clip(cut, -1.0, 1.0) * 32767).astype(np.int16)
        import wave
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm16.tobytes())
        out.append(out_path)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("eval_data/no_rhythm/commons_longform"))
    parser.add_argument("--short-dest", type=Path, default=Path("eval_data/no_rhythm/commons_longform_5s"))
    args = parser.parse_args()

    print(f"Checking license/duration for {len(CANDIDATE_TITLES)} candidates...")
    files = fetch_metadata(CANDIDATE_TITLES)
    print(f"{len(files)}/{len(CANDIDATE_TITLES)} passed the license+metadata check")
    for f in files:
        print(f"  {f.duration_s:>8.1f}s  {f.license:<12}  {f.title}")

    print(f"\nDownloading + converting to {args.dest}...")
    long_paths = [download_and_convert(f, args.dest) for f in files]

    print(f"Building matched 5s prefixes in {args.short_dest} for a same-content duration comparison...")
    short_paths = build_short_variants(long_paths, args.short_dest)

    print(f"\nDone: {len(long_paths)} long clips, {len(short_paths)} matched 5s clips")
