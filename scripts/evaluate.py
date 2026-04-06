"""
evaluate.py
-----------
Backtest evaluation on the configured test season(s).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from features.feature_pipeline import FEATURE_COLS, TARGET_COL
from models.ensemble import EnsembleModel


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate(cfg_path: str = "config/config.yaml") -> None:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data"]["processed_path"], parse_dates=["date"])
    test_df = df[df["season"].isin(cfg["data"]["test_seasons"])].copy()
    print(f"Test set: {len(test_df)} matches  (seasons {cfg['data']['test_seasons']})\n")

    ensemble = EnsembleModel.load("models/ensemble.pkl")

    preds_df = ensemble.predict(test_df[FEATURE_COLS])
    probs = preds_df[["p_home_win", "p_draw", "p_away_win"]].values
    y_true = test_df[TARGET_COL].values
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, probs)
    brier = brier_score_multiclass(y_true, probs)

    baseline_probs = np.full((len(y_true), 3), 1e-9)
    baseline_probs[:, 0] = 1.0 - 2e-9
    base_acc = accuracy_score(y_true, np.zeros(len(y_true), dtype=int))
    base_ll = log_loss(y_true, baseline_probs)

    print("=" * 55)
    print(f"{'Metric':<25} {'Model':>12} {'Baseline':>12}")
    print("-" * 55)
    print(f"{'Accuracy':<25} {acc:>12.3f} {base_acc:>12.3f}")
    print(f"{'Log-Loss':<25} {ll:>12.4f} {base_ll:>12.4f}")
    print(f"{'Brier Score':<25} {brier:>12.4f}")
    print("=" * 55)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual H", "Actual D", "Actual A"],
        columns=["Pred H", "Pred D", "Pred A"],
    )
    print(cm_df.to_string())

    all_seasons = cfg["data"]["train_seasons"] + cfg["data"]["val_seasons"] + cfg["data"]["test_seasons"]
    print("\n\nBacktest by season:")
    print(f"{'Season':<10} {'N':>6} {'Accuracy':>10} {'LogLoss':>10}")
    print("-" * 40)
    for season in all_seasons:
        s_df = df[df["season"] == season].copy()
        s_probs_df = ensemble.predict(s_df[FEATURE_COLS])
        s_probs = s_probs_df[["p_home_win", "p_draw", "p_away_win"]].values
        s_y = s_df[TARGET_COL].values
        s_acc = accuracy_score(s_y, s_probs.argmax(axis=1))
        s_ll = log_loss(s_y, s_probs)
        tag = " <- test" if season in cfg["data"]["test_seasons"] else (" <- val" if season in cfg["data"]["val_seasons"] else "")
        print(f"{season:<10} {len(s_df):>6} {s_acc:>10.3f} {s_ll:>10.4f}{tag}")

    print("\n\nCalibration check (home-win probability bins):")
    fraction_pos, mean_predicted = calibration_curve((y_true == 0).astype(int), probs[:, 0], n_bins=5)
    print(f"{'Predicted':>12} {'Actual':>12}")
    for mp, fp in zip(mean_predicted, fraction_pos):
        print(f"{mp:>12.3f} {fp:>12.3f}")


if __name__ == "__main__":
    evaluate()
