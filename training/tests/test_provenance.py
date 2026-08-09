from beat_recognizer_training.data.provenance import ALLOWED_LICENSES, classify_cc_url


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


def test_garbage_and_empty_rejected():
    assert classify_cc_url("") is None
    assert classify_cc_url(None) is None
    assert classify_cc_url("not a url at all") is None
    assert classify_cc_url("https://example.com/some/other/license") is None


def test_every_accepted_value_is_on_the_allowlist():
    urls = [
        "http://creativecommons.org/licenses/by/3.0/",
        "http://creativecommons.org/licenses/by/4.0/",
        "http://creativecommons.org/licenses/by-sa/3.0/",
        "http://creativecommons.org/licenses/by-sa/4.0/",
        "https://creativecommons.org/publicdomain/zero/1.0/",
    ]
    for url in urls:
        result = classify_cc_url(url)
        assert result is not None
        assert result in ALLOWED_LICENSES, f"{url} -> {result} not in ALLOWED_LICENSES"
