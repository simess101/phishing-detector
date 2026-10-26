"""Char-CNN tests. Skipped automatically where PyTorch isn't installed (like CI)."""

import numpy as np
import pytest

from phishdetect.charcnn import PAD, UNK, CharVocab

torch = pytest.importorskip("torch")

from phishdetect.charcnn import CharCNNClassifier  # noqa: E402

LEGIT = ["https://www.vcu.edu/admissions/graduate", "https://github.com/a/b/blob/main/readme.md",
         "https://en.wikipedia.org/wiki/phishing", "https://docs.python.org/3/library/re.html"]
PHISH = ["http://paypal-verify.xyz/login.php?id=1", "https://secure-apple-id.top/signin/verify.php",
         "http://bank-update.click/wp-includes/js/auth.php", "https://x7k2q.pages.dev/office365/login.html"]
URLS = (LEGIT + PHISH) * 20
Y = np.tile([0] * 4 + [1] * 4, 20)


def test_vocab_pads_and_marks_unknown():
    v = CharVocab.build(["abc", "abd"], min_count=1)
    enc = v.encode(["ab", "az"], max_len=4)
    assert enc.shape == (2, 4)
    assert enc[0, 2] == PAD and enc[1, 1] == UNK


def test_trains_and_scores_on_cpu():
    m = CharCNNClassifier(filters=16, embed_dim=8).fit(URLS, Y, epochs=8, batch_size=16,
                                                       device=torch.device("cpu"), verbose=False)
    p = m.phish_proba(LEGIT + PHISH)
    assert p.shape == (8,) and ((p >= 0) & (p <= 1)).all()
    assert p[4:].mean() > p[:4].mean()          # learned something on this toy set


def test_early_stopping_keeps_history():
    m = CharCNNClassifier(filters=8, embed_dim=8).fit(URLS, Y, val_urls=LEGIT + PHISH, val_y=[0] * 4 + [1] * 4,
                                                      epochs=3, batch_size=32, device=torch.device("cpu"),
                                                      verbose=False)
    assert 1 <= len(m.history) <= 3 and "val_pr_auc" in m.history[0]


def test_pickle_roundtrip(tmp_path):
    import joblib
    m = CharCNNClassifier(filters=8, embed_dim=8).fit(URLS, Y, epochs=2, batch_size=32,
                                                      device=torch.device("cpu"), verbose=False)
    joblib.dump(m, tmp_path / "m.joblib")
    m2 = joblib.load(tmp_path / "m.joblib")
    m2.device = m.device = torch.device("cpu")
    assert np.allclose(m.phish_proba(URLS[:8]), m2.phish_proba(URLS[:8]), atol=1e-6)


def test_empty_input():
    m = CharCNNClassifier(filters=8, embed_dim=8).fit(URLS, Y, epochs=1, device=torch.device("cpu"), verbose=False)
    assert m.phish_proba([]).shape == (0,)
