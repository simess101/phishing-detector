from phishdetect.features import FEATURE_NAMES, extract_features, featurize


def test_returns_every_feature():
    feats = extract_features("https://www.example.com/path?q=1")
    assert list(feats) == FEATURE_NAMES


def test_https_and_ip():
    assert extract_features("https://example.com")["has_https"] == 1
    assert extract_features("http://192.168.10.5/login")["has_ip"] == 1
    assert extract_features("example.com")["has_ip"] == 0


def test_missing_scheme_still_parses_host():
    feats = extract_features("paypal.com.secure-login.xyz/verify")
    assert feats["host_len"] == len("paypal.com.secure-login.xyz")
    assert feats["n_subdomains"] == 2
    assert feats["n_suspicious_words"] >= 2


def test_shortener_and_at_sign():
    assert extract_features("https://bit.ly/3abc")["is_shortener"] == 1
    assert extract_features("http://google.com@evil.ru")["has_at"] == 1


def test_bad_port_does_not_crash():
    assert extract_features("http://example.com:abc/")["has_port"] == 1


def test_featurize_shape():
    df = featurize(["a.com", "https://b.org/x", "http://1.2.3.4"])
    assert df.shape == (3, len(FEATURE_NAMES))


from phishdetect.features import host_only, normalize_url


def test_normalize_strips_scheme_and_www():
    assert normalize_url("https://www.PayPal.com/login") == "PayPal.com/login"
    assert normalize_url("http://www2.example.org") == "example.org"
    assert normalize_url("example.com/a") == "example.com/a"
    # only a leading www. is removed
    assert normalize_url("http://login.www.evil.ru") == "login.www.evil.ru"


def test_host_only():
    assert host_only("https://www.paypal.com/login?x=1") == "paypal.com"
    assert host_only("secure-login.xyz/verify") == "secure-login.xyz"
    assert host_only("http://192.168.1.5:8080/a") == "192.168.1.5"
