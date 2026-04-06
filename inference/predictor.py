"""
predictor.py
------------
Inference layer that wraps the ensemble model and the feature pipeline.

At prediction time we don't have a real feature row (the match hasn't been
played). Instead we reconstruct a feature vector from:
  - The latest Elo ratings for both teams (loaded from saved state)
  - The latest rolling stats for both teams (loaded from saved state)

This module exposes a single Predictor class used by the FastAPI app.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from typing import Any

from features.feature_pipeline import FEATURE_COLS
from models.ensemble import EnsembleModel


class Predictor:
    """
    Loads pre-trained ensemble and latest team-level feature snapshots.
    """

    def __init__(self, cfg_path: str = "config/config.yaml"):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.ensemble = EnsembleModel.load("models/ensemble.pkl")

        # Load the latest feature snapshot (last row per team from processed data)
        self._team_snapshots = self._build_team_snapshots()
        self._teams = sorted(self._team_snapshots.keys())

    # ------------------------------------------------------------------ #
    def _build_team_snapshots(self) -> dict[str, dict]:
        """
        For each team, extract their latest feature values from the full
        processed dataset. These serve as the 'current state' for predictions.
        """
        df = pd.read_csv(self.cfg["data"]["processed_path"], parse_dates=["date"])
        snapshots: dict[str, dict] = {}

        for _, row in df.sort_values("date").iterrows():
            # Update home team snapshot
            home = row["home_team"]
            snapshots[home] = {
                "elo": row["home_elo"],
                "rolling_scored": row["home_rolling_scored"],
                "rolling_conceded": row["home_rolling_conceded"],
                "win_rate": row["home_win_rate"],
                "draw_rate": row["home_draw_rate"],
                "loss_rate": row["home_loss_rate"],
                "form_pts": row["home_form_pts"],
                "venue_scored": row["home_venue_scored"],
                "venue_conceded": row["home_venue_conceded"],
                "attack_strength": row["home_attack_strength"],
                "defence_strength": row["home_defence_strength"],
            }
            # Update away team snapshot
            away = row["away_team"]
            snapshots[away] = {
                "elo": row["away_elo"],
                "rolling_scored": row["away_rolling_scored"],
                "rolling_conceded": row["away_rolling_conceded"],
                "win_rate": row["away_win_rate"],
                "draw_rate": row["away_draw_rate"],
                "loss_rate": row["away_loss_rate"],
                "form_pts": row["away_form_pts"],
                "venue_scored": row["away_venue_scored"],
                "venue_conceded": row["away_venue_conceded"],
                "attack_strength": row["away_attack_strength"],
                "defence_strength": row["away_defence_strength"],
            }
        return snapshots

    # ------------------------------------------------------------------ #
    def _build_feature_row(
        self,
        home_team: str,
        away_team: str,
    ) -> pd.DataFrame:
        """Construct a single-row feature DataFrame for the given matchup."""
        h = self._team_snapshots.get(home_team, {})
        a = self._team_snapshots.get(away_team, {})

        # Default to league-average values if team is unknown
        default_elo = 1500.0
        default_rate = 1.3
        default_stats = 0.33

        row: dict[str, Any] = {
            # Elo
            "home_elo": h.get("elo", default_elo),
            "away_elo": a.get("elo", default_elo),
            # Rolling goals
            "home_rolling_scored":   h.get("rolling_scored", default_rate),
            "home_rolling_conceded": h.get("rolling_conceded", default_rate),
            "away_rolling_scored":   a.get("rolling_scored", default_rate),
            "away_rolling_conceded": a.get("rolling_conceded", default_rate),
            # Form rates
            "home_win_rate":  h.get("win_rate", default_stats),
            "home_draw_rate": h.get("draw_rate", default_stats),
            "home_loss_rate": h.get("loss_rate", default_stats),
            "home_form_pts":  h.get("form_pts", 1.0),
            "away_win_rate":  a.get("win_rate", default_stats),
            "away_draw_rate": a.get("draw_rate", default_stats),
            "away_loss_rate": a.get("loss_rate", default_stats),
            "away_form_pts":  a.get("form_pts", 1.0),
            # Venue
            "home_venue_scored":    h.get("venue_scored", default_rate),
            "home_venue_conceded":  h.get("venue_conceded", default_rate),
            "away_venue_scored":    a.get("venue_scored", default_rate),
            "away_venue_conceded":  a.get("venue_conceded", default_rate),
            # H2H — set to neutral if not available
            "home_h2h_scored":    1.3,
            "home_h2h_conceded":  1.3,
            "home_h2h_win_rate":  0.33,
            "away_h2h_scored":    1.3,
            "away_h2h_conceded":  1.3,
            "away_h2h_win_rate":  0.33,
            # Strength
            "home_attack_strength":  h.get("attack_strength", 1.0),
            "home_defence_strength": h.get("defence_strength", 1.0),
            "away_attack_strength":  a.get("attack_strength", 1.0),
            "away_defence_strength": a.get("defence_strength", 1.0),
        }
        row["elo_diff"] = row["home_elo"] - row["away_elo"]
        return pd.DataFrame([row])

    # ------------------------------------------------------------------ #
    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Full prediction for a given matchup.

        Returns
        -------
        dict with keys:
            home_team, away_team,
            p_home_win, p_draw, p_away_win,
            expected_home_goals, expected_away_goals,
            top_scorelines: list[dict{scoreline, probability}]
        """
        X = self._build_feature_row(home_team, away_team)
        result = self.ensemble.predict(X).iloc[0]

        # Top 10 most likely scorelines
        scorelines = result["scorelines"]
        top_sl = sorted(scorelines.items(), key=lambda kv: kv[1], reverse=True)[:10]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "p_home_win": round(float(result["p_home_win"]), 4),
            "p_draw":     round(float(result["p_draw"]), 4),
            "p_away_win": round(float(result["p_away_win"]), 4),
            "expected_home_goals": round(float(result["lam_home"]), 2),
            "expected_away_goals": round(float(result["lam_away"]), 2),
            "top_scorelines": [
                {"scoreline": sl, "probability": round(p, 4)}
                for sl, p in top_sl
            ],
        }

    @property
    def available_teams(self) -> list[str]:
        return self._teams
