from beat_recognizer_training.data.fetch_jamendo import JamendoTrack, classify_license


def make_track(license_ccurl: str, **kwargs) -> JamendoTrack:
    return JamendoTrack(
        track_id=kwargs.get("track_id", "1"),
        name=kwargs.get("name", "test"),
        license_ccurl=license_ccurl,
        audiodownload=kwargs.get("audiodownload", "https://example.com/a.mp3"),
        audiodownload_allowed=kwargs.get("audiodownload_allowed", True),
    )


def test_attribution_accepted():
    t = make_track("http://creativecommons.org/licenses/by/3.0/")
    assert classify_license(t) == "CC-BY-3.0"


def test_attribution_sharealike_accepted():
    t = make_track("http://creativecommons.org/licenses/by-sa/4.0/")
    assert classify_license(t) == "CC-BY-SA-4.0"


def test_noncommercial_rejected():
    t = make_track("http://creativecommons.org/licenses/by-nc-nd/3.0/")
    assert classify_license(t) is None


def test_empty_license_rejected():
    t = make_track("")
    assert classify_license(t) is None
