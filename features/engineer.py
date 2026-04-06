"""
features/engineer.py
--------------------
Orchestrates all feature engineering steps and produces the final feature matrix.

Pipeline
────────
1. Load raw match data
2. Compute rolling form features (causal, per-team)
3. Compute ELO ratings (causal, pre-match)
4. Add attack/defense strength relative to league average
5. Add home/away split features
6. Encode target variable
7. Save processed feature DataFrame

Design: Every feature is computed using only information available *before*
the match kicks off.  The pipeline is deterministic given the raw CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from features.elo import EloSystem
from features.form import compute_form_features


# ── Attack / Defense strength ──────────────────────────────────────────────────

def _compute_attack_defense_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute attack/defense strength of each team
    relative to the league average *for that season* (computed from
    matches already played, so strictly causal).

    attack_strength  = goals_scored_per_game  / league_avg_goals_per_game
    defense_strength = goals_conceded_per_game / league_avg_goals_per_game
    (> 1 means stronger than average)
    """
    df = df.copy()
    df["home_attack_strength"]  = np.nan
    df["home_defense_strength"] = np.nan
    df["away_attack_strength"]  = np.nan
    df["away_defense_strength"] = np.nan

    for idx, row in df.iterrows():
        season     = row["season"]
        match_date = row["date"]
        past       = df[(df["season"] == season) & (df["date"] < match_date)]

        if past.empty:
            # No prior data — default to 1.0 (league average)
            league_avg = 1.35
        else:
            total_g = past["home_goals"].sum() + past["away_goals"].sum()
            total_m = len(past)
            league_avg = (total_g / (total_m * 2)) if total_m else 1.35

        def team_strength(team: str) -> tuple[float, float]:
            as_home = past[past["home_team"] == team]
            as_away = past[past["away_team"] == team]
            gf = list(as_home["home_goals"]) + list(as_away["away_goals"])
            ga = list(as_home["away_goals"]) + list(as_away["home_goals"])
            if not gf:
                return 1.0, 1.0
            atk = np.mean(gf) / league_avg if league_avg else 1.0
            dfc = np.mean(ga) / league_avg if league_avg else 1.0
            return float(atk), float(dfc)

        home_atk, home_dfc = team_strength(row["home_team"])
        away_atk, away_dfc = team_strength(row["away_team"])

        df.at[idx, "home_attack_strength"]  = home_atk
        df.at[idx, "home_defense_strength"] = home_dfc
        df.at[idx, "away_attack_strength"]  = away_atk
        df.at[idx, "away_defense_strength"] = away_dfc

    return df


# ── Differential features ──────────────────────────────────────────────────────

def _add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise differences between home and away features —
    often more predictive than raw values.
    """
    df = df.copy()
    df["form_pts_diff"]   = df["home_form_pts_avg"]  - df["away_form_pts_avg"]
    df["form_gd_diff"]    = df["home_form_gd_avg"]   - df["away_form_gd_avg"]
    df["form_xg_diff"]    = df["home_form_xg_avg"]   - df["away_form_xg_avg"]
    df["atk_str_diff"]    = df["home_attack_strength"]  - df["away_attack_strength"]
    df["def_str_diff"]    = df["home_defense_strength"] - df["away_defense_strength"]
    df["elo_diff"]        = df["elo_home_pre"] - df["elo_away_pre"]
    return df


# ── Target encoding ────────────────────────────────────────────────────────────

def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    result: H=0, D=1, A=2   (home team perspective)
    Also keep home_goals / away_goals as regression targets for Poisson model.
    """
    df = df.copy()
    result_map = {"H": 0, "D": 1, "A": 2}
    df["target"] = df["result"].map(result_map)
    return df


# ── Feature column list ────────────────────────────────────────────────────────

FEATURE_COLS = [
    # ELO
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "elo_prob_home", "elo_prob_draw", "elo_prob_away",
    # Rolling form — home
    "home_form_pts_avg", "home_form_gf_avg", "home_form_ga_avg",
    "home_form_gd_avg", "home_form_shots_avg", "home_form_xg_avg",
    "home_form_win_rate", "home_form_draw_rate", "home_form_loss_rate",
    "home_form_pts_decayed", "home_form_gf_decayed", "home_form_ga_decayed",
    # Rolling form — away
    "away_form_pts_avg", "away_form_gf_avg", "away_form_ga_avg",
    "away_form_gd_avg", "away_form_shots_avg", "away_form_xg_avg",
    "away_form_win_rate", "away_form_draw_rate", "away_form_loss_rate",
    "away_form_pts_decayed", "away_form_gf_decayed", "away_form_ga_decayed",
    # Attack / defense strength
    "home_attack_strength", "home_defense_strength",
    "away_attack_strength", "away_defense_strength",
    # Head-to-head
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
    "h2h_home_goals", "h2h_away_goals",
    # Differentials
    "form_pts_diff", "form_gd_diff", "form_xg_diff",
    "atk_str_diff", "def_str_diff",
]

TARGET_COL    = "target"
POISSON_COLS  = ["home_goals", "away_goals"]   # regression targets


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_features(
    raw_path: str | Path,
    processed_path: str | Path,
    form_window: int  = 5,
    half_life: float  = 60.0,
    h2h_window: int   = 10,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    raw_path        : path to raw CSV
    processed_path  : where to save the feature DataFrame
    form_window     : rolling window size for form features
    half_life       : days for time-decay weighting
    h2h_window      : head-to-head history size

    Returns
    -------
    DataFrame with all features and targets, sorted by date.
    """
    print("─" * 60)
    print("Feature Engineering Pipeline")
    print("─" * 60)

    # 1. Load
    print("[ 1/5 ] Loading raw data …")
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"        {len(df):,} matches loaded")

    # 2. Rolling form
    print("[ 2/5 ] Computing rolling form features …")
    df = compute_form_features(df, window=form_window, half_life=half_life,
                               h2h_window=h2h_window)

    # 3. ELO ratings
    print("[ 3/5 ] Computing ELO ratings …")
    elo = EloSystem()
    df = elo.process_dataframe(df)

    # 4. Attack / Defense strength
    print("[ 4/5 ] Computing attack/defense strength …")
    df = _compute_attack_defense_strength(df)

    # 5. Differentials + target
    print("[ 5/5 ] Adding differential features and encoding target …")
    df = _add_differential_features(df)
    df = _encode_target(df)

    # Validate feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    # Save
    out_path = Path(processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved feature matrix → {out_path}")
    print(f"  Shape: {df.shape}  |  Features: {len(FEATURE_COLS)}")
    print("─" * 60)

    return df


def load_features(processed_path: str | Path) -> pd.DataFrame:
    """Load pre-computed feature CSV."""
    df = pd.read_csv(processed_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)
