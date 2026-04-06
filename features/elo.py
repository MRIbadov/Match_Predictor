"""
elo.py
------
Implements an Elo rating system from scratch, updated match-by-match in
chronological order so ratings at prediction time never include future data.

Formula recap
-------------
Expected score for team A vs B:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

After match:
    R_A_new = R_A + K * (S_A - E_A)

where S_A ∈ {1, 0.5, 0} for win / draw / loss.

Home advantage is modelled by adding `home_advantage` points to the home
team's rating before computing expected scores, then removing it afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class EloSystem:
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    home_advantage: float = 100.0
    decay_factor: float = 0.95  # regress toward mean between seasons

    ratings: Dict[str, float] = field(default_factory=dict)
    _current_season: int = field(default=-1, repr=False)

    # ------------------------------------------------------------------ #
    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """E_A: probability that A wins (draw counts 0.5 for both)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _apply_season_decay(self, season: int) -> None:
        """Regress all ratings toward the mean at the start of a new season."""
        mean = np.mean(list(self.ratings.values())) if self.ratings else self.initial_rating
        self.ratings = {
            t: mean + self.decay_factor * (r - mean)
            for t, r in self.ratings.items()
        }

    # ------------------------------------------------------------------ #
    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        season: int,
    ) -> Tuple[float, float]:
        """
        Update ratings for one match. Returns (home_elo_before, away_elo_before)
        — i.e., ratings *before* the match, suitable for feature use.
        """
        if season != self._current_season:
            if self._current_season != -1:
                self._apply_season_decay(season)
            self._current_season = season

        # Initialise unknown teams
        for team in (home_team, away_team):
            if team not in self.ratings:
                self.ratings[team] = self.initial_rating

        r_home = self.ratings[home_team]
        r_away = self.ratings[away_team]

        # Snapshot *before* update — these become features
        r_home_pre = r_home
        r_away_pre = r_away

        # Apply home advantage for expected-score calculation only
        e_home = self._expected_score(r_home + self.home_advantage, r_away)
        e_away = 1.0 - e_home

        # Actual scores
        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # Update
        self.ratings[home_team] = r_home + self.k_factor * (s_home - e_home)
        self.ratings[away_team] = r_away + self.k_factor * (s_away - e_away)

        return r_home_pre, r_away_pre


# ------------------------------------------------------------------ #
def compute_elo_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Iterate over all matches in chronological order and attach pre-match
    Elo ratings as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw match data, must contain columns:
        date, season, home_team, away_team, home_goals, away_goals
    cfg : dict
        elo sub-section of config.yaml

    Returns
    -------
    pd.DataFrame with additional columns:
        home_elo, away_elo, elo_diff
    """
    elo = EloSystem(
        initial_rating=cfg.get("initial_rating", 1500),
        k_factor=cfg.get("k_factor", 32),
        home_advantage=cfg.get("home_advantage", 100),
        decay_factor=cfg.get("decay_factor", 0.95),
    )

    df = df.sort_values("date").reset_index(drop=True)
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h_elo, a_elo = elo.update(
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_goals=int(row["home_goals"]),
            away_goals=int(row["away_goals"]),
            season=int(row["season"]),
        )
        home_elos.append(h_elo)
        away_elos.append(a_elo)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df
