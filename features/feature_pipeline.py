"""
feature_pipeline.py
-------------------
Orchestrates the active feature engineering pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.elo import compute_elo_features
from features.rolling_stats import RollingStatsEngine, compute_strength_features


FEATURE_COLS = [
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

TARGET_COL = "outcome"
POISSON_TARGETS = ["home_goals", "away_goals"]


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"H": 0, "D": 1, "A": 2}
    df[TARGET_COL] = df["result"].map(mapping)
    return df


def impute_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
        for season in df["season"].unique():
            mask = df["season"] == season
            median_val = df.loc[mask, col].median()
            df.loc[mask & df[col].isna(), col] = median_val
    return df


def build_features(cfg_path: str = "config/config.yaml") -> pd.DataFrame:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = cfg["data"]["raw_path"]
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path, parse_dates=["date"])

    print("Computing Elo ratings...")
    df = compute_elo_features(df, cfg["elo"])

    print("Computing rolling team statistics...")
    engine = RollingStatsEngine(
        window=cfg["features"]["rolling_window"],
        half_life=cfg["features"]["time_decay_half_life"],
        min_matches=cfg["features"]["min_matches_for_form"],
    )
    df = engine.process(df)

    print("Computing attack/defence strength...")
    df = compute_strength_features(df)

    df = encode_target(df)
    df = impute_missing(df, FEATURE_COLS)

    out_path = Path(cfg["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Feature matrix saved -> {out_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    build_features()
