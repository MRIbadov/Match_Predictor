"""
poisson_model.py
----------------
Bivariate Poisson-style goal prediction using two regularized regressors.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POISSON_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_attack_strength",
    "home_defence_strength",
    "away_attack_strength",
    "away_defence_strength",
    "home_venue_scored",
    "home_venue_conceded",
    "away_venue_scored",
    "away_venue_conceded",
    "home_rolling_scored",
    "home_rolling_conceded",
    "away_rolling_scored",
    "away_rolling_conceded",
]


class PoissonGoalModel:
    """Fits two Ridge regression models: one for home goals and one for away goals."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model_home = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        self._model_away = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        self.features = POISSON_FEATURES
        self._fitted = False

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series) -> "PoissonGoalModel":
        X_mat = X[self.features].values
        self._model_home.fit(X_mat, np.log(y_home + 0.5))
        self._model_away.fit(X_mat, np.log(y_away + 0.5))
        self._fitted = True
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_mat = X[self.features].values
        lam_home = np.exp(self._model_home.predict(X_mat)) - 0.5
        lam_away = np.exp(self._model_away.predict(X_mat)) - 0.5
        lam_home = np.clip(lam_home, 0.1, 8.0)
        lam_away = np.clip(lam_away, 0.1, 8.0)
        return lam_home, lam_away

    @staticmethod
    def scoreline_matrix(lam_home: float, lam_away: float, max_goals: int = 6) -> np.ndarray:
        home_probs = poisson.pmf(np.arange(max_goals + 1), lam_home)
        away_probs = poisson.pmf(np.arange(max_goals + 1), lam_away)
        return np.outer(home_probs, away_probs)

    @staticmethod
    def outcome_probs_from_matrix(mat: np.ndarray) -> Tuple[float, float, float]:
        p_home = float(np.sum(np.tril(mat, k=-1)))
        p_draw = float(np.trace(mat))
        p_away = float(np.sum(np.triu(mat, k=1)))
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def predict_outcome_probs(self, X: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
        lam_h, lam_a = self.predict_lambdas(X)
        rows = []
        for lh, la in zip(lam_h, lam_a):
            mat = self.scoreline_matrix(lh, la, max_goals)
            ph, pd_, pa = self.outcome_probs_from_matrix(mat)
            rows.append(
                {
                    "p_home_win": ph,
                    "p_draw": pd_,
                    "p_away_win": pa,
                    "lam_home": lh,
                    "lam_away": la,
                }
            )
        return pd.DataFrame(rows, index=X.index)

    def save(self, path: str = "models/poisson_model.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Poisson model saved -> {path}")

    @classmethod
    def load(cls, path: str = "models/poisson_model.pkl") -> "PoissonGoalModel":
        with open(path, "rb") as f:
            return pickle.load(f)
