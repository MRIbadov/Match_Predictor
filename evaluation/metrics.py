"""
evaluation/metrics.py
---------------------
Model evaluation toolkit for multi-class football outcome prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss


def _clip_probs(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0 - eps)
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / row_sums


def calibration_curve_data(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data for a single class.

    Returns (mean_predicted_prob, fraction_of_positives, bin_counts).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred = []
    frac_pos = []
    counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true_bin[mask].mean())
        counts.append(mask.sum())

    return np.array(mean_pred), np.array(frac_pos), np.array(counts)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Lower is better; 0 indicates perfect calibration.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    eps: float = 1e-7,
) -> dict[str, Any]:
    """
    Compute evaluation metrics for a multi-class probabilistic predictor.
    """
    probs = _clip_probs(y_prob, eps)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, probs)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    brier_scores = {}
    ece_scores = {}
    for cls_idx, cls_name in [(0, "home_win"), (1, "draw"), (2, "away_win")]:
        y_bin = (y_true == cls_idx).astype(int)
        brier_scores[cls_name] = float(brier_score_loss(y_bin, probs[:, cls_idx]))
        ece_scores[cls_name] = expected_calibration_error(y_bin, probs[:, cls_idx])

    rps = _ranked_probability_score(y_true, probs)

    return {
        "model": model_name,
        "n_samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier_home": round(brier_scores["home_win"], 4),
        "brier_draw": round(brier_scores["draw"], 4),
        "brier_away": round(brier_scores["away_win"], 4),
        "brier_mean": round(np.mean(list(brier_scores.values())), 4),
        "ece_home": round(ece_scores["home_win"], 4),
        "ece_draw": round(ece_scores["draw"], 4),
        "ece_away": round(ece_scores["away_win"], 4),
        "ece_mean": round(np.mean(list(ece_scores.values())), 4),
        "rps_mean": round(rps, 4),
        "confusion_matrix": cm.tolist(),
    }


def _ranked_probability_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Ranked probability score for ordinal three-class outcomes.
    """
    k = 3
    rps_total = 0.0
    for i in range(len(y_true)):
        outcome = int(y_true[i])
        cdf_true = np.array([(1 if j >= outcome else 0) for j in range(k)], dtype=float)
        cdf_pred = np.cumsum(y_prob[i])
        rps_total += np.sum((cdf_pred - cdf_true) ** 2) / (k - 1)
    return rps_total / len(y_true)


def baseline_always_home(n: int) -> np.ndarray:
    probs = np.zeros((n, 3))
    probs[:, 0] = 1.0
    return probs


def baseline_uniform(n: int) -> np.ndarray:
    return np.full((n, 3), 1.0 / 3.0)


def baseline_historical(y_train: np.ndarray, n: int) -> np.ndarray:
    probs_const = np.array([
        (y_train == 0).mean(),
        (y_train == 1).mean(),
        (y_train == 2).mean(),
    ])
    return np.tile(probs_const, (n, 1))


def backtest_by_season(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    seasons: list[int],
) -> pd.DataFrame:
    """
    Evaluate model performance separately per season.
    """
    target_col = "target" if "target" in df.columns else "outcome"
    rows = []
    for season in seasons:
        mask = df["season"] == season
        if mask.sum() == 0:
            continue
        y_true = df.loc[mask, target_col].values
        probs = y_prob[mask]
        metrics = evaluate_predictions(y_true, probs, model_name=str(season))
        metrics["season"] = season
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("season")


def print_report(metrics_list: list[dict]) -> None:
    """
    Print a formatted comparison table of multiple models.
    """
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Accuracy':>10} {'LogLoss':>10} {'Brier':>10} {'ECE':>10} {'RPS':>10}")
    print("-" * 70)
    for metrics in metrics_list:
        print(
            f"{metrics['model']:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['log_loss']:>10.4f} "
            f"{metrics['brier_mean']:>10.4f} "
            f"{metrics['ece_mean']:>10.4f} "
            f"{metrics['rps_mean']:>10.4f}"
        )
    print("=" * 70)
    print("Lower is better for LogLoss, Brier, ECE, RPS.")
    print()

    if metrics_list and "confusion_matrix" in metrics_list[0]:
        cm = np.array(metrics_list[0]["confusion_matrix"])
        print(f"Confusion Matrix ({metrics_list[0]['model']}):")
        print(f"{'':12} {'Pred:H':>8} {'Pred:D':>8} {'Pred:A':>8}")
        for label, row in zip(["True:H", "True:D", "True:A"], cm):
            print(f"  {label:<10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
        print()
