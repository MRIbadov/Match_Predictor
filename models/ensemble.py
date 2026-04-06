"""
ensemble.py
-----------
Blend Poisson goal probabilities with the classifier probabilities.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from models.classifier import OutcomeClassifier
from models.poisson_model import PoissonGoalModel


class EnsembleModel:
    def __init__(
        self,
        poisson_model: PoissonGoalModel,
        classifier: OutcomeClassifier,
        poisson_weight: float = 0.4,
        xgb_weight: float = 0.6,
        max_scoreline: int = 6,
    ):
        assert abs(poisson_weight + xgb_weight - 1.0) < 1e-6, "Weights must sum to 1"
        self.poisson = poisson_model
        self.classifier = classifier
        self.w_poisson = poisson_weight
        self.w_xgb = xgb_weight
        self.max_scoreline = max_scoreline

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p_poisson = self.poisson.predict_outcome_probs(X, self.max_scoreline)
        p_clf = self.classifier.predict_outcome_probs(X)

        blended = pd.DataFrame(
            {
                "p_home_win": self.w_poisson * p_poisson["p_home_win"] + self.w_xgb * p_clf["p_home_win"],
                "p_draw": self.w_poisson * p_poisson["p_draw"] + self.w_xgb * p_clf["p_draw"],
                "p_away_win": self.w_poisson * p_poisson["p_away_win"] + self.w_xgb * p_clf["p_away_win"],
                "lam_home": p_poisson["lam_home"],
                "lam_away": p_poisson["lam_away"],
            },
            index=X.index,
        )

        lam_h, lam_a = self.poisson.predict_lambdas(X)
        scorelines_list = []
        for lh, la in zip(lam_h, lam_a):
            mat = PoissonGoalModel.scoreline_matrix(lh, la, self.max_scoreline)
            scorelines = {
                f"{i}-{j}": float(mat[i, j])
                for i in range(self.max_scoreline + 1)
                for j in range(self.max_scoreline + 1)
            }
            scorelines_list.append(scorelines)
        blended["scorelines"] = scorelines_list
        return blended

    def save(self, path: str = "models/ensemble.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Ensemble saved -> {path}")

    @classmethod
    def load(cls, path: str = "models/ensemble.pkl") -> "EnsembleModel":
        with open(path, "rb") as f:
            return pickle.load(f)
