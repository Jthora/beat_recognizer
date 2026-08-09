from beat_recognizer_training.data.fetch_fma import FmaTrack, classify_license


def test_attribution_with_confirmable_url_accepted():
    track = FmaTrack(
        track_id="2",
        raw_license="Attribution",
        license_url="http://creativecommons.org/licenses/by/3.0/",
    )
    assert classify_license(track) == "CC-BY-3.0"


def test_attribution_without_url_rejected_not_guessed():
    # No license_url to confirm the CC version from — must not be assumed.
    track = FmaTrack(track_id="3", raw_license="Attribution", license_url=None)
    assert classify_license(track) is None


def test_cc0_plain_text_accepted_even_without_url():
    track = FmaTrack(track_id="4", raw_license="Creative Commons 0", license_url=None)
    assert classify_license(track) == "CC0-1.0"


def test_noncommercial_rejected_regardless_of_url():
    track = FmaTrack(
        track_id="5",
        raw_license="Attribution-NonCommercial-ShareAlike",
        license_url="http://creativecommons.org/licenses/by-nc-sa/3.0/",
    )
    assert classify_license(track) is None


def test_noderivatives_rejected():
    track = FmaTrack(
        track_id="6",
        raw_license="Attribution-NoDerivatives",
        license_url="http://creativecommons.org/licenses/by-nd/3.0/",
    )
    assert classify_license(track) is None


def test_blank_license_rejected():
    track = FmaTrack(track_id="7", raw_license="", license_url=None)
    assert classify_license(track) is None


def test_url_overrides_ambiguous_text_when_it_disagrees():
    # If the URL says NC even though the plain text alone wouldn't obviously
    # say so, the URL (authoritative) wins and the track is rejected.
    track = FmaTrack(
        track_id="8",
        raw_license="Some Custom License Name",
        license_url="http://creativecommons.org/licenses/by-nc/4.0/",
    )
    assert classify_license(track) is None


def test_version_parsed_directly_from_fma_plain_text():
    # FMA's real tracks.csv has no license_url column at all (confirmed
    # against the actual downloaded file) — this is the real path, not a
    # fallback. FMA spells the version out in the license string itself.
    track = FmaTrack(track_id="9", raw_license="Attribution 3.0 International", license_url=None)
    assert classify_license(track) == "CC-BY-3.0"


def test_sharealike_variant_spacing_from_plain_text():
    track = FmaTrack(track_id="10", raw_license="Attribution-Share Alike 3.0 United States", license_url=None)
    assert classify_license(track) == "CC-BY-SA-3.0"


def test_older_cc_versions_accepted_not_just_3_and_4():
    # CC licenses are irrevocable once applied — CC BY 2.0 is exactly as
    # valid today as CC BY 4.0, so this must not be rejected just because
    # it's an older version.
    track = FmaTrack(track_id="11", raw_license="Attribution 2.0 France", license_url=None)
    assert classify_license(track) == "CC-BY-2.0"


def test_space_separated_no_derivative_works_rejected():
    # Regression test: FMA phrases some ND licenses as "No Derivative
    # Works" (spaced), which the contiguous "noderiv"/"no-deriv" substring
    # checks missed, wrongly accepting 59 real ND-licensed tracks in a
    # verification run against actual FMA metadata before this was fixed.
    track = FmaTrack(
        track_id="13",
        raw_license="Attribution-No Derivative Works 3.0 United States",
        license_url=None,
    )
    assert classify_license(track) is None


def test_public_domain_common_law_recording_rejected():
    # "Public Domain" text present, but the common-law-recording-rights
    # caveat means it isn't actually unambiguous.
    track = FmaTrack(
        track_id="12",
        raw_license="Public Domain / Sound Recording Common Law Protection",
        license_url=None,
    )
    assert classify_license(track) is None
