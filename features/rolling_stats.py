"""
rolling_stats.py
----------------
Computes team-level rolling statistics with time-decay weighting.

Features computed per team per match (using only past matches):
  - Rolling goals scored / conceded (last N matches)
  - Rolling win / draw / loss rate
  - Home vs away splits
  - Head-to-head stats between the two specific teams
  - Attack / defence strength relative to league average
  - Time-decay weighted versions of the above

Time-decay formula
------------------
For a match played `delta_days` before the current match:
    weight = exp(-ln(2) * delta_days / half_life)

This gives weight=1 to a match played today and weight=0.5 to one played
`half_life` days ago, following the radioactive-decay analogy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


HALF_LIFE_DAYS = 90  # configurable via cfg


def _decay_weight(delta_days: float, half_life: float) -> float:
    """Exponential decay weight given age in days."""
    return np.exp(-np.log(2) * delta_days / half_life)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if weights.sum() == 0:
        return np.nan
    return float(np.sum(values * weights) / weights.sum())


# ------------------------------------------------------------------ #
class RollingStatsEngine:
    """
    Processes matches in chronological order. For each match it computes
    backward-looking statistics from the perspective of *both* teams.
    """

    def __init__(self, window: int = 5, half_life: float = 90.0, min_matches: int = 3):
        self.window = window
        self.half_life = half_life
        self.min_matches = min_matches
        # history[team] = list of dicts with past match info
        self._history: dict[str, list] = {}

    # ------------------------------------------------------------------ #
    def _get_team_history(self, team: str) -> list:
        return self._history.get(team, [])

    def _compute_team_features(
        self,
        team: str,
        current_date: pd.Timestamp,
        is_home: bool,
        opponent: Optional[str] = None,
    ) -> dict:
        hist = self._get_team_history(team)
        if not hist:
            return self._empty_features(team, is_home)

        df_hist = pd.DataFrame(hist)
        df_hist["age_days"] = (current_date - df_hist["date"]).dt.days.clip(lower=0)
        df_hist["weight"] = df_hist["age_days"].apply(
            lambda d: _decay_weight(d, self.half_life)
        )

        recent = df_hist.tail(self.window)
        recent_w = recent["weight"].values

        # ---- Basic rolling metrics ------------------------------------ #
        goals_scored = recent["goals_scored"].values
        goals_conceded = recent["goals_conceded"].values
        wins = (recent["result_for_team"] == "W").astype(float).values
        draws = (recent["result_for_team"] == "D").astype(float).values
        losses = (recent["result_for_team"] == "L").astype(float).values

        n = len(recent)
        ok = n >= self.min_matches  # enough history?

        prefix = "home" if is_home else "away"
        feats: dict = {}

        feats[f"{prefix}_rolling_scored"] = _weighted_mean(goals_scored, recent_w) if ok else np.nan
        feats[f"{prefix}_rolling_conceded"] = _weighted_mean(goals_conceded, recent_w) if ok else np.nan
        feats[f"{prefix}_win_rate"] = _weighted_mean(wins, recent_w) if ok else np.nan
        feats[f"{prefix}_draw_rate"] = _weighted_mean(draws, recent_w) if ok else np.nan
        feats[f"{prefix}_loss_rate"] = _weighted_mean(losses, recent_w) if ok else np.nan
        feats[f"{prefix}_form_pts"] = _weighted_mean(
            recent["points"].values, recent_w
        ) if ok else np.nan

        # ---- Home / away split --------------------------------------- #
        venue_mask = recent["was_home"] == is_home
        if venue_mask.sum() >= self.min_matches:
            sub = recent[venue_mask]
            sub_w = sub["weight"].values
            feats[f"{prefix}_venue_scored"] = _weighted_mean(sub["goals_scored"].values, sub_w)
            feats[f"{prefix}_venue_conceded"] = _weighted_mean(sub["goals_conceded"].values, sub_w)
        else:
            feats[f"{prefix}_venue_scored"] = feats[f"{prefix}_rolling_scored"]
            feats[f"{prefix}_venue_conceded"] = feats[f"{prefix}_rolling_conceded"]

        # ---- Head-to-head ------------------------------------------- #
        if opponent:
            h2h = df_hist[df_hist["opponent"] == opponent]
            if len(h2h) >= 1:
                hw = h2h["weight"].values
                feats[f"{prefix}_h2h_scored"] = _weighted_mean(h2h["goals_scored"].values, hw)
                feats[f"{prefix}_h2h_conceded"] = _weighted_mean(h2h["goals_conceded"].values, hw)
                feats[f"{prefix}_h2h_win_rate"] = _weighted_mean(
                    (h2h["result_for_team"] == "W").astype(float).values, hw
                )
            else:
                feats[f"{prefix}_h2h_scored"] = np.nan
                feats[f"{prefix}_h2h_conceded"] = np.nan
                feats[f"{prefix}_h2h_win_rate"] = np.nan
        return feats

    def _empty_features(self, team: str, is_home: bool) -> dict:
        prefix = "home" if is_home else "away"
        return {
            f"{prefix}_rolling_scored": np.nan,
            f"{prefix}_rolling_conceded": np.nan,
            f"{prefix}_win_rate": np.nan,
            f"{prefix}_draw_rate": np.nan,
            f"{prefix}_loss_rate": np.nan,
            f"{prefix}_form_pts": np.nan,
            f"{prefix}_venue_scored": np.nan,
            f"{prefix}_venue_conceded": np.nan,
            f"{prefix}_h2h_scored": np.nan,
            f"{prefix}_h2h_conceded": np.nan,
            f"{prefix}_h2h_win_rate": np.nan,
        }

    # ------------------------------------------------------------------ #
    def _update_history(self, match: pd.Series) -> None:
        """Store both teams' perspective of this match for future lookups."""
        home, away = match["home_team"], match["away_team"]
        hg, ag = int(match["home_goals"]), int(match["away_goals"])
        date = match["date"]

        if hg > ag:
            h_res, a_res = "W", "L"
            h_pts, a_pts = 3, 0
        elif ag > hg:
            h_res, a_res = "L", "W"
            h_pts, a_pts = 0, 3
        else:
            h_res = a_res = "D"
            h_pts = a_pts = 1

        for team, opp, gs, gc, res, pts, was_home in [
            (home, away, hg, ag, h_res, h_pts, True),
            (away, home, ag, hg, a_res, a_pts, False),
        ]:
            self._history.setdefault(team, []).append({
                "date": date,
                "opponent": opp,
                "goals_scored": gs,
                "goals_conceded": gc,
                "result_for_team": res,
                "points": pts,
                "was_home": was_home,
            })

    # ------------------------------------------------------------------ #
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry: iterate chronologically, compute features *before*
        updating history (prevents data leakage).
        """
        df = df.sort_values("date").reset_index(drop=True)
        all_feats = []

        for _, row in df.iterrows():
            home_feats = self._compute_team_features(
                row["home_team"], row["date"], is_home=True, opponent=row["away_team"]
            )
            away_feats = self._compute_team_features(
                row["away_team"], row["date"], is_home=False, opponent=row["home_team"]
            )
            all_feats.append({**home_feats, **away_feats})
            self._update_history(row)  # update AFTER reading features

        feat_df = pd.DataFrame(all_feats, index=df.index)
        return pd.concat([df, feat_df], axis=1)


# ------------------------------------------------------------------ #
def compute_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attack / defence strength relative to the rolling league average.

    attack_strength  = team_goals_scored  / league_avg_goals_scored
    defence_strength = team_goals_conceded / league_avg_goals_conceded

    These are computed per-season to avoid mixing different eras.
    """
    records = []
    for season, grp in df.groupby("season"):
        league_avg_scored = grp["home_goals"].mean() + grp["away_goals"].mean()
        league_avg_scored /= 2  # per team per match

        for _, row in grp.iterrows():
            home_atk = row.get("home_rolling_scored", np.nan) / league_avg_scored \
                if not pd.isna(row.get("home_rolling_scored", np.nan)) else np.nan
            away_atk = row.get("away_rolling_scored", np.nan) / league_avg_scored \
                if not pd.isna(row.get("away_rolling_scored", np.nan)) else np.nan
            home_def = row.get("home_rolling_conceded", np.nan) / league_avg_scored \
                if not pd.isna(row.get("home_rolling_conceded", np.nan)) else np.nan
            away_def = row.get("away_rolling_conceded", np.nan) / league_avg_scored \
                if not pd.isna(row.get("away_rolling_conceded", np.nan)) else np.nan
            records.append({
                "home_attack_strength": home_atk,
                "away_attack_strength": away_atk,
                "home_defence_strength": home_def,
                "away_defence_strength": away_def,
            })

    strength_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, strength_df], axis=1)
