"""Model builders shared across notebooks.

Every model here takes raw URL strings and returns P(phishing) from
`phish_proba(urls)`. Each one does its own preprocessing (normalizing,
cutting down to the hostname), so callers can score them all on the same
benchmark without knowing how each one sees a URL.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .features import featurize, featurize_host, host_only, normalize_url

PATH_FEATURES = ["path_len", "query_len", "has_double_slash_path"]


def raw_url(url: str) -> str:
    """The URL unchanged. A named function (not a lambda) so models can be saved with joblib."""
    return url


class FeatureForest:
    """Random forest on hand-built features, computed from one 'view' of the URL.

    featurizer is featurize (19 lexical features, notebooks 01 to 03) or
    featurize_host (PSL-based host features, notebook 04).
    """

    def __init__(self, view=raw_url, drop=(), random_state=42, featurizer=featurize,
                 estimator=None, **rf_kwargs):
        self.view = view
        self.drop = list(drop)
        self.featurizer = featurizer
        if estimator is None:
            params = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state)
            params.update(rf_kwargs)
            estimator = RandomForestClassifier(**params)
        self.model = estimator

    def _features(self, urls):
        return self.featurizer([self.view(str(u)) for u in urls]).drop(columns=self.drop)

    def fit(self, urls, y):
        X = self._features(urls)
        self.columns_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def phish_proba(self, urls):
        return self.model.predict_proba(self._features(urls))[:, 1]

    def feature_importances(self) -> pd.Series:
        """Random forest only; boosted models don't expose this directly."""
        return pd.Series(self.model.feature_importances_, index=self.columns_).sort_values()


class HostCharNgram:
    """TF-IDF over 3 to 5 character chunks of the hostname, then logistic regression."""

    def __init__(self, ngram_range=(3, 5), min_df=3, max_features=200_000, C=4.0):
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df,
                            sublinear_tf=True, max_features=max_features),
            LogisticRegression(max_iter=2000, C=C),
        )

    def _hosts(self, urls):
        return [host_only(str(u)) for u in urls]

    def fit(self, urls, y):
        self.model.fit(self._hosts(urls), y)
        return self

    def phish_proba(self, urls):
        return self.model.predict_proba(self._hosts(urls))[:, 1]

    def top_ngrams(self, k=25) -> tuple[list[str], list[str]]:
        """(most phishy, most legit) n-grams by weight."""
        vec, clf = self.model[0], self.model[-1]
        w = pd.Series(clf.coef_[0], index=vec.get_feature_names_out())
        return list(w.nlargest(k).index), list(w.nsmallest(k).index)


def build_models(random_state=42) -> dict:
    """The four experiments from notebook 02, untrained."""
    return {
        "A · original": FeatureForest(raw_url, [], random_state),
        "B · no scheme": FeatureForest(normalize_url, ["has_https"], random_state),
        "C · host only": FeatureForest(host_only, ["has_https"] + PATH_FEATURES, random_state),
        "D · host char n-grams": HostCharNgram(),
    }


def host_psl_forest(random_state=42) -> FeatureForest:
    """Model E: random forest on the PSL-based host features."""
    return FeatureForest(host_only, [], random_state, featurizer=featurize_host)


def host_psl_boosted(random_state=42) -> FeatureForest:
    """Model E with gradient boosting instead of a random forest.

    Same features, but the saved model is a few MB instead of hundreds, which
    matters once it ships inside a Docker image.
    """
    est = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=random_state)
    return FeatureForest(host_only, [], random_state, featurizer=featurize_host, estimator=est)