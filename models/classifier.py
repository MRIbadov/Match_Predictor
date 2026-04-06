"""
classifier.py
-------------
XGBoost multi-class classifier for match outcome prediction.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss


CLASSIFIER_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_rolling_scored",
    "home_rolling_conceded",
    "away_rolling_scored",
    "away_rolling_conceded",
    "home_win_rate",
    "home_draw_rate",
    "home_loss_rate",
    "home_form_pts",
    "away_win_rate",
    "away_draw_rate",
    "away_loss_rate",
    "away_form_pts",
    "home_venue_scored",
    "home_venue_conceded",
    "away_venue_scored",
    "away_venue_conceded",
    "home_h2h_scored",
    "home_h2h_conceded",
    "home_h2h_win_rate",
    "away_h2h_scored",
    "away_h2h_conceded",
    "away_h2h_win_rate",
    "home_attack_strength",
    "home_defence_strength",
    "away_attack_strength",
    "away_defence_strength",
]


class OutcomeClassifier:
    def __init__(self, xgb_params: dict | None = None):
        params = xgb_params or {}
        base = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 3),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
        )
        self._clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.features = CLASSIFIER_FEATURES
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "OutcomeClassifier":
        self._clf.fit(X_train[self.features], y_train)
        self._fitted = True

        if X_val is not None and y_val is not None:
            probs = self.predict_proba(X_val)
            acc = accuracy_score(y_val, probs.argmax(axis=1))
            ll = log_loss(y_val, probs)
            print(f"  Validation accuracy: {acc:.3f}  |  log-loss: {ll:.4f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(X[self.features])

    def predict_outcome_probs(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(X)
        return pd.DataFrame(
            {
                "p_home_win": probs[:, 0],
                "p_draw": probs[:, 1],
                "p_away_win": probs[:, 2],
            },
            index=X.index,
        )

    def feature_importances(self) -> pd.Series:
        importances = np.zeros(len(self.features))
        for est in self._clf.calibrated_classifiers_:
            importances += est.estimator.feature_importances_
        importances /= len(self._clf.calibrated_classifiers_)
        return pd.Series(importances, index=self.features).sort_values(ascending=False)

    def save(self, path: str = "models/classifier.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Classifier saved -> {path}")

    @classmethod
    def load(cls, path: str = "models/classifier.pkl") -> "OutcomeClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
