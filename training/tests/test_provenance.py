from beat_recognizer_training.data.provenance import ALLOWED_LICENSES, classify_cc_url, is_allowed


def test_attribution_urls_accepted():
    assert classify_cc_url("http://creativecommons.org/licenses/by/3.0/") == "CC-BY-3.0"
    assert classify_cc_url("https://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"


def test_attribution_sharealike_accepted():
    assert classify_cc_url("http://creativecommons.org/licenses/by-sa/3.0/") == "CC-BY-SA-3.0"
    assert classify_cc_url("https://creativecommons.org/licenses/by-sa/4.0/") == "CC-BY-SA-4.0"


def test_public_domain_zero_accepted():
    assert classify_cc_url("https://creativecommons.org/publicdomain/zero/1.0/") == "CC0-1.0"


def test_noncommercial_variants_rejected():
    assert classify_cc_url("http://creativecommons.org/licenses/by-nc/3.0/") is None
    assert classify_cc_url("http://creativecommons.org/licenses/by-nc-sa/4.0/") is None
    assert classify_cc_url("http://creativecommons.org/licenses/by-nc-nd/3.0/") is None


def test_noderivatives_variants_rejected():
    assert classify_cc_url("http://creativecommons.org/licenses/by-nd/3.0/") is None
    assert classify_cc_url("http://creativecommons.org/licenses/by-nc-nd/4.0/") is None


def test_free_art_license_accepted():
    # Regression: seen live in Jamendo data (33 tracks in a 500-track
    # sample) and silently rejected by omission before this was added,
    # despite Free Art License already being on ALLOWED_LICENSES.
    assert classify_cc_url("http://artlibre.org/licence/lal/") == "FREE-ART-LICENSE"
    assert classify_cc_url("https://artlibre.org/licence/lal/1.3/") == "FREE-ART-LICENSE"


def test_garbage_and_empty_rejected():
    assert classify_cc_url("") is None
    assert classify_cc_url(None) is None
    assert classify_cc_url("not a url at all") is None
    assert classify_cc_url("https://example.com/some/other/license") is None


def test_every_accepted_value_passes_is_allowed():
    urls = [
        "http://creativecommons.org/licenses/by/3.0/",
        "http://creativecommons.org/licenses/by/4.0/",
        "http://creativecommons.org/licenses/by-sa/3.0/",
        "http://creativecommons.org/licenses/by-sa/4.0/",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "http://artlibre.org/licence/lal/",
    ]
    for url in urls:
        result = classify_cc_url(url)
        assert result is not None
        assert is_allowed(result), f"{url} -> {result} not accepted by is_allowed"


def test_unusual_jurisdiction_port_version_accepted():
    # Regression: a real Australian "CC-BY-SA-2.1" track crashed a fetch
    # script whose validation only recognized the five common version
    # numbers — permission scope is set by by/nc/nd/sa, not the version
    # number, so any version of a by/by-sa license must be accepted.
    assert classify_cc_url("http://creativecommons.org/licenses/by-sa/2.1/au/") == "CC-BY-SA-2.1"
    assert is_allowed("CC-BY-SA-2.1")
    assert is_allowed("CC-BY-2.1")
    assert "CC-BY-SA-2.1" not in ALLOWED_LICENSES  # not enumerated, but still allowed
    # Still must not accept made-up non-CC categories via the version regex.
    assert not is_allowed("CC-BY-NC-3.0")
    assert not is_allowed("Whatever-3.0")
