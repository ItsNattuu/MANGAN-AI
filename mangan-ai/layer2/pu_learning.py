"""
Positive-Unlabeled (PU) learning via bagging.

The problem: there are no reliable negatives. A random background point
might simply be an UNDISCOVERED deposit. Treating unlabeled cells as
y=0 teaches the model "manganese exists nowhere except these N pixels",
which is both wrong and useless for exploration.

The fix: repeatedly sample a small pool of unlabeled points as
*pseudo-negatives*, train a model each time, and average. Any single
unlabeled cell is only a pseudo-negative in a fraction of rounds, so
genuinely prospective ground still scores high on average.

Bonus: the standard deviation across rounds is a free uncertainty
estimate. Mean = the map. Std = confidence in the map.
"""

import numpy as np
from sklearn.base import clone


class BaggingPU:
    """Bagging-based Positive-Unlabeled classifier.

    Parameters
    ----------
    base_estimator : sklearn-compatible classifier
        Typically XGBClassifier or RandomForestClassifier.
    n_estimators : int
        Number of bagging rounds (K).
    random_state : int
    """

    def __init__(self, base_estimator, n_estimators=100, random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        """X : array (n_samples, n_features)
        y : array where 1 = known positive, 0 = UNLABELED (not negative)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        pos_idx = np.where(y == 1)[0]
        unl_idx = np.where(y == 0)[0]
        n_pos = len(pos_idx)

        if n_pos == 0:
            raise ValueError("No positive samples found.")
        if len(unl_idx) < n_pos:
            raise ValueError("Fewer unlabeled points than positives.")

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for k in range(self.n_estimators):
            # sample pseudo-negatives, same count as positives
            sampled_unl = rng.choice(unl_idx, size=n_pos, replace=False)
            idx = np.concatenate([pos_idx, sampled_unl])

            X_k = X[idx]
            y_k = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])

            est = clone(self.base_estimator)
            est.fit(X_k, y_k)
            self.estimators_.append(est)

        return self

    def predict_proba_mean_std(self, X):
        """Returns (mean_score, std_score) across all K estimators.

        mean -> prospectivity raster
        std  -> uncertainty raster
        """
        X = np.asarray(X)
        preds = np.column_stack([
            est.predict_proba(X)[:, 1] for est in self.estimators_
        ])
        return preds.mean(axis=1), preds.std(axis=1)

    def predict_proba(self, X):
        """sklearn-compatible interface. Returns (n, 2) array."""
        mean, _ = self.predict_proba_mean_std(X)
        return np.column_stack([1 - mean, mean])

    def feature_importances_mean(self):
        """Average feature importance across the ensemble, if the base
        estimator exposes feature_importances_."""
        importances = []
        for est in self.estimators_:
            if hasattr(est, "feature_importances_"):
                importances.append(est.feature_importances_)
        if not importances:
            return None
        return np.mean(importances, axis=0)
