"""Corpus-diversity addition: pulls the Freesound Loop Dataset (9,455 loops,
Ramires et al. ISMIR 2020) as a genuinely *confirmed*-tempo training signal,
distinct from every other source in this project's corpus -- Jamendo/FMA
tracks are labeled by our own DSP+ensemble pipeline (an estimate, even a
cross-checked one), while these loops' BPM comes from the producer's own
upload-time tag, with ~3,000 of the 9,455 additionally expert-verified
(discard/well_cut/bpm fields) by the dataset's own annotators -- a two-stage
confirmation, not an algorithmic guess.

Real caveat, stated plainly: these are short, repetitive loops, not full
songs. Valuable for hardening the tempo-estimation objective specifically
(clean, unambiguous single-tempo ground truth), not a substitute for the
corpus's full-song structural/genre diversity.

Same per-item license discipline as fetch_jamendo.py/fetch_fma.py: only
CC0/CC-BY/CC-BY-SA clear the bar (spec §8), reusing provenance.py's
classify_cc_url so the classification logic isn't duplicated a third time.

Source: Zenodo record 3967852. The *compilation* is CC BY 4.0, but that
covers the dataset packaging, not the individual sounds -- per-sound audio
license is read from FSL10K/metadata.json and checked per-file exactly like
every other source, never assumed from the compilation's license. No
Freesound API key needed: audio ships bundled in the archive (confirmed
directly), not fetched per-track via the API. Zenodo doesn't honor HTTP
Range requests on this file (confirmed: a ranged GET still returns 200 with
the full Content-Length, not 206) so there's no way to selectively fetch
only the license-clean subset -- the full ~8.8GB archive has to download
before filtering happens locally.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import requests

from beat_recognizer_training.data import provenance

FSL_ZENODO_RECORD = "3967852"
FSL_ZIP_URL = f"https://zenodo.org/api/records/{FSL_ZENODO_RECORD}/files/FSL10K.zip/content"
FSL_ANNOTATIONS_URL = f"https://zenodo.org/api/records/{FSL_ZENODO_RECORD}/files/annotations.zip/content"


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"{dest} already exists, skipping download")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        next_report = 100 << 20  # every 100MB
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    print(f"  {written / 1e6:.0f} / {total / 1e6:.0f} MB")
                    next_report += 100 << 20
    tmp.rename(dest)


def load_expert_annotations(annotations_zip: Path) -> dict[str, dict]:
    """sound_id -> expert annotation dict, for whichever ~3,000 of the
    9,455 sounds have one (see this module's docstring)."""
    out = {}
    with zipfile.ZipFile(annotations_zip) as zf:
        for name in zf.namelist():
            stem = Path(name).stem
            if not stem.startswith("sound-"):
                continue
            sound_id = stem.removeprefix("sound-")
            out[sound_id] = json.loads(zf.read(name))
    return out


def index_audio_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    """sound_id -> archive path. Filenames are `audio/wav/<sound_id>_<revision_id>.wav.wav`
    -- confirmed directly (not the flat `<sound_id>.wav` the dataset's own
    README implies) -- so sound_id has to be recovered from the filename
    prefix, not constructed from it."""
    out = {}
    for name in zf.namelist():
        if not name.startswith("audio/wav/") or not name.endswith(".wav.wav"):
            continue
        sound_id = Path(name).name.split("_")[0]
        out[sound_id] = name
    return out


def import_loops(
    zip_path: Path,
    annotations_zip: Path,
    dest_dir: Path,
    manifest_path: Path,
    labels_path: Path,
    limit: int | None = None,
) -> None:
    """Writes two things: the usual PROVENANCE.jsonl entry every training
    source gets, plus a labels_path record with the confirmed BPM itself --
    provenance.ProvenanceEntry has no bpm field (it's a shared schema, not
    worth widening for one source), and without this the actual point of
    pulling this dataset (real ground truth, not another estimate) would
    never get persisted anywhere. This is meant to feed Phase 4's training
    set directly, bypassing dsp_teacher/ensemble's bootstrap-and-cross-check
    entirely -- there's nothing to estimate or corroborate here."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    expert = load_expert_annotations(annotations_zip)
    already = provenance.load_logged_paths(manifest_path)
    print(f"{len(expert)} expert-annotated sounds loaded, {len(already)} already logged")

    kept = skipped_license = skipped_no_tempo = skipped_discarded = skipped_no_audio = 0
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_file = labels_path.open("a")

    with zipfile.ZipFile(zip_path) as zf:
        metadata = json.loads(zf.read("metadata.json"))
        audio_paths = index_audio_paths(zf)
        sound_ids = list(metadata.keys())
        if limit:
            sound_ids = sound_ids[:limit]

        for i, sound_id in enumerate(sound_ids):
            info = metadata[sound_id]
            license_str = provenance.classify_cc_url(info.get("license"))
            if license_str is None:
                skipped_license += 1
                continue

            expert_info = expert.get(sound_id)
            if expert_info is not None:
                if expert_info.get("discard") or not expert_info.get("defined_tempo"):
                    skipped_discarded += 1
                    continue
                try:
                    bpm = float(expert_info["bpm"])
                except (KeyError, ValueError, TypeError):
                    skipped_no_tempo += 1
                    continue
                bpm_source = "expert"
            else:
                # No expert pass for this sound -- fall back to the
                # producer-tag-derived BPM the dataset construction already
                # parsed (see this module's docstring; the paper reports
                # ~5% error/ambiguity on this tier).
                raw_bpm = info.get("annotations", {}).get("bpm")
                if not raw_bpm:
                    skipped_no_tempo += 1
                    continue
                try:
                    bpm = float(raw_bpm)
                except (ValueError, TypeError):
                    skipped_no_tempo += 1
                    continue
                bpm_source = "producer_tag"

            archive_path = audio_paths.get(sound_id)
            if archive_path is None:
                skipped_no_audio += 1
                continue

            dest_path = dest_dir / f"{sound_id}.wav"
            if str(dest_path) not in already:
                if not dest_path.exists():
                    with zf.open(archive_path) as src, dest_path.open("wb") as dst:
                        dst.write(src.read())
                provenance.append_entry(
                    manifest_path,
                    file_path=dest_path,
                    source_url=f"https://freesound.org/s/{sound_id}/",
                    license=license_str,
                    source_dataset="freesound-loops",
                )
                labels_file.write(
                    json.dumps(
                        {"audio_path": str(dest_path), "bpm": bpm, "bpm_source": bpm_source, "sound_id": sound_id}
                    )
                    + "\n"
                )
                labels_file.flush()
            kept += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(sound_ids)} (kept {kept})")

    labels_file.close()
    print(
        f"\nKept {kept}, skipped {skipped_license} (license), "
        f"{skipped_discarded} (expert-flagged discard/no-tempo), "
        f"{skipped_no_tempo} (no usable bpm field), {skipped_no_audio} (no matching audio file)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("freesound_loops"))
    parser.add_argument("--dest", type=Path, default=Path("freesound_loops/audio"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("../data/PROVENANCE.jsonl"),
        help="the same manifest every other training source logs to -- one audit trail, not a per-source one",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("../data/FREESOUND_LOOPS_LABELS.jsonl"),
        help="confirmed-bpm ground truth, meant to feed Phase 4 directly (no dsp_teacher/ensemble bootstrap needed)",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    zip_path = args.archive_dir / "FSL10K.zip"
    annotations_zip = args.archive_dir / "annotations.zip"
    download_file(FSL_ZIP_URL, zip_path)
    download_file(FSL_ANNOTATIONS_URL, annotations_zip)
    import_loops(zip_path, annotations_zip, args.dest, args.manifest, args.labels, limit=args.limit)
