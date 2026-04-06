"""
scripts/train.py
----------------
Train the full football prediction stack using the current feature/model APIs.

Run from project root:
    python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from data_ingestion.sync import sync_official_matches
from evaluation.metrics import (
    backtest_by_season,
    baseline_always_home,
    baseline_historical,
    baseline_uniform,
    evaluate_predictions,
    print_report,
)
from features.feature_pipeline import FEATURE_COLS, TARGET_COL, build_features
from models.classifier import OutcomeClassifier
from models.ensemble import EnsembleModel
from models.poisson_model import PoissonGoalModel


def load_config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_aware_split(
    df: pd.DataFrame,
    train_seasons: list[int],
    val_seasons: list[int],
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(train_seasons)].copy()
    val = df[df["season"].isin(val_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()

    print(f"  Train: {len(train):,} matches (seasons {train_seasons})")
    print(f"  Val:   {len(val):,} matches (seasons {val_seasons})")
    print(f"  Test:  {len(test):,} matches (seasons {test_seasons})")
    return train, val, test


def main() -> None:
    cfg = load_config()

    print("\n" + "=" * 60)
    print("STEP 0: Official Data Sync")
    print("=" * 60)
    sync_result = sync_official_matches(cfg)
    print(f"  Window: {sync_result.date_from} -> {sync_result.date_to}")
    print(f"  Matches fetched: {sync_result.matches_fetched}")
    if sync_result.reason:
        print(f"  Note: {sync_result.reason}")

    print("\n" + "=" * 60)
    print("STEP 1: Feature Engineering")
    print("=" * 60)

    feat_path = Path(cfg["data"]["processed_path"])
    if sync_result.changed and feat_path.exists():
        feat_path.unlink()
        print(f"Removed stale cached features at {feat_path} after official data sync.")

    if feat_path.exists():
        print(f"Loading cached features from {feat_path} ...")
        df = pd.read_csv(feat_path, parse_dates=["date"])
    else:
        df = build_features("config/config.yaml")

    missing = [col for col in FEATURE_COLS + [TARGET_COL, "home_goals", "away_goals"] if col not in df.columns]
    if missing:
        raise ValueError(f"Feature data is missing required columns: {missing}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "home_goals", "away_goals"]).reset_index(drop=True)
    print(f"  Feature matrix: {df.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Time-Aware Data Split")
    print("=" * 60)

    train_df, val_df, test_df = time_aware_split(
        df,
        train_seasons=cfg["data"]["train_seasons"],
        val_seasons=cfg["data"]["val_seasons"],
        test_seasons=cfg["data"]["test_seasons"],
    )

    y_train = train_df[TARGET_COL]
    y_val = val_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    print("\n" + "=" * 60)
    print("STEP 3: Poisson Goal Model")
    print("=" * 60)

    poisson = PoissonGoalModel(alpha=cfg["models"]["poisson"].get("alpha", 1.0))
    poisson.fit(train_df, train_df["home_goals"], train_df["away_goals"])

    pois_val_df = poisson.predict_outcome_probs(val_df)
    p_pois_val = pois_val_df[["p_home_win", "p_draw", "p_away_win"]].values
    m_pois = evaluate_predictions(y_val.values, p_pois_val, "Poisson")
    print(f"  Val accuracy: {m_pois['accuracy']:.4f}  |  Log-loss: {m_pois['log_loss']:.4f}")

    print("\n" + "=" * 60)
    print("STEP 4: XGBoost Classifier")
    print("=" * 60)

    classifier = OutcomeClassifier(xgb_params=cfg["models"].get("xgboost", {}))
    classifier.fit(train_df, y_train, val_df, y_val)

    clf_val_df = classifier.predict_outcome_probs(val_df)
    p_clf_val = clf_val_df[["p_home_win", "p_draw", "p_away_win"]].values
    m_clf = evaluate_predictions(y_val.values, p_clf_val, "XGBoost")
    print(f"  Val accuracy: {m_clf['accuracy']:.4f}  |  Log-loss: {m_clf['log_loss']:.4f}")

    feature_importances = classifier.feature_importances()
    print("\n  Top 10 features:")
    print(feature_importances.head(10).to_string())

    print("\n" + "=" * 60)
    print("STEP 5: Ensemble")
    print("=" * 60)

    ensemble = EnsembleModel(
        poisson_model=poisson,
        classifier=classifier,
        poisson_weight=cfg["models"]["ensemble"]["poisson_weight"],
        xgb_weight=cfg["models"]["ensemble"]["xgboost_weight"],
        max_scoreline=cfg["api"].get("max_scoreline", 6),
    )
    ens_val = ensemble.predict(val_df)
    p_ens_val = ens_val[["p_home_win", "p_draw", "p_away_win"]].values
    m_ens_val = evaluate_predictions(y_val.values, p_ens_val, "Ensemble (val)")
    print(f"  Val accuracy: {m_ens_val['accuracy']:.4f}  |  Log-loss: {m_ens_val['log_loss']:.4f}")

    print("\n" + "=" * 60)
    print("STEP 6: Test Set Evaluation")
    print("=" * 60)

    p_pois_test = poisson.predict_outcome_probs(test_df)[["p_home_win", "p_draw", "p_away_win"]].values
    p_clf_test = classifier.predict_outcome_probs(test_df)[["p_home_win", "p_draw", "p_away_win"]].values
    p_ens_test = ensemble.predict(test_df)[["p_home_win", "p_draw", "p_away_win"]].values

    n_test = len(test_df)
    metrics = [
        evaluate_predictions(y_test.values, baseline_always_home(n_test), "Baseline:AlwaysHome"),
        evaluate_predictions(y_test.values, baseline_uniform(n_test), "Baseline:Uniform"),
        evaluate_predictions(y_test.values, baseline_historical(y_train.values, n_test), "Baseline:Historical"),
        evaluate_predictions(y_test.values, p_pois_test, "Poisson GLM"),
        evaluate_predictions(y_test.values, p_clf_test, "XGBoost"),
        evaluate_predictions(y_test.values, p_ens_test, "Ensemble"),
    ]
    print_report(metrics)

    print("Per-season backtest (ensemble):")
    bt_df = pd.concat([val_df, test_df]).reset_index(drop=True)
    bt_probs = np.vstack([p_ens_val, p_ens_test])
    bt = backtest_by_season(
        bt_df,
        bt_probs,
        seasons=cfg["data"]["val_seasons"] + cfg["data"]["test_seasons"],
    )
    print(bt[["accuracy", "log_loss", "rps_mean"]].to_string())

    print("\n" + "=" * 60)
    print("STEP 7: Saving Models")
    print("=" * 60)

    Path("models").mkdir(exist_ok=True)
    poisson.save("models/poisson_model.pkl")
    classifier.save("models/classifier.pkl")
    ensemble.save("models/ensemble.pkl")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
