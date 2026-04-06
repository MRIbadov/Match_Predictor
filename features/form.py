"""
features/form.py
----------------
Computes rolling form features for each team:
  • Points per game (last N matches)
  • Goals scored / conceded rolling average
  • Shots on target rolling average
  • Win / draw / loss ratio
  • Time-decay weighted versions of the above

All features are computed in a strictly causal manner — only past matches
are used so there is zero data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Time-decay weight ─────────────────────────────────────────────────────────

def time_decay_weight(days_ago: float, half_life: float = 60.0) -> float:
    """
    Exponential decay: weight = 0.5^(days_ago / half_life).
    A match played `half_life` days ago is worth half as much.
    """
    return 0.5 ** (days_ago / half_life)


# ── Per-team match history builder ────────────────────────────────────────────

def _build_team_records(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Decompose match-level DataFrame into a per-team list of match records.
    Each record contains metrics from that team's perspective.
    """
    team_history: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        for perspective in ("home", "away"):
            if perspective == "home":
                team      = row["home_team"]
                opp       = row["away_team"]
                gf        = row["home_goals"]
                ga        = row["away_goals"]
                shots     = row.get("home_shots", np.nan)
                xg        = row.get("home_xg", np.nan)
                result    = row["result"]          # H / D / A
                pts       = 3 if result == "H" else (1 if result == "D" else 0)
                is_home   = 1
            else:
                team      = row["away_team"]
                opp       = row["home_team"]
                gf        = row["away_goals"]
                ga        = row["home_goals"]
                shots     = row.get("away_shots", np.nan)
                xg        = row.get("away_xg", np.nan)
                result    = row["result"]
                pts       = 3 if result == "A" else (1 if result == "D" else 0)
                is_home   = 0

            record = {
                "match_idx": int(row.name),
                "date":      row["date"],
                "team":      team,
                "opponent":  opp,
                "gf":        gf,
                "ga":        ga,
                "shots":     shots,
                "xg":        xg,
                "pts":       pts,
                "is_home":   is_home,
            }

            team_history.setdefault(team, []).append(record)

    return team_history


# ── Rolling form computer ─────────────────────────────────────────────────────

def _rolling_features(
    history: list[dict],
    match_date: pd.Timestamp,
    window: int,
    half_life: float,
) -> dict[str, float]:
    """
    Given a team's match history up to (exclusive) match_date,
    compute rolling and decay-weighted features from the last `window` matches.
    """
    past = [r for r in history if r["date"] < match_date]
    past = past[-window:]          # keep last N (already sorted by date)

    if not past:
        return {
            "form_pts_avg":     0.0,
            "form_gf_avg":      0.0,
            "form_ga_avg":      0.0,
            "form_gd_avg":      0.0,
            "form_shots_avg":   0.0,
            "form_xg_avg":      0.0,
            "form_win_rate":    0.0,
            "form_draw_rate":   0.0,
            "form_loss_rate":   0.0,
            "form_pts_decayed": 0.0,
            "form_gf_decayed":  0.0,
            "form_ga_decayed":  0.0,
            "n_recent":         0,
        }

    days_ago_list = [(match_date - r["date"]).days for r in past]
    weights       = np.array([time_decay_weight(d, half_life) for d in days_ago_list])
    w_sum         = weights.sum() or 1.0

    pts_arr   = np.array([r["pts"]  for r in past], dtype=float)
    gf_arr    = np.array([r["gf"]   for r in past], dtype=float)
    ga_arr    = np.array([r["ga"]   for r in past], dtype=float)
    shots_arr = np.array([r["shots"] for r in past], dtype=float)
    xg_arr    = np.array([r["xg"]   for r in past], dtype=float)

    n = len(past)

    return {
        "form_pts_avg":     float(pts_arr.mean()),
        "form_gf_avg":      float(gf_arr.mean()),
        "form_ga_avg":      float(ga_arr.mean()),
        "form_gd_avg":      float((gf_arr - ga_arr).mean()),
        "form_shots_avg":   float(np.nanmean(shots_arr)),
        "form_xg_avg":      float(np.nanmean(xg_arr)),
        "form_win_rate":    float((pts_arr == 3).mean()),
        "form_draw_rate":   float((pts_arr == 1).mean()),
        "form_loss_rate":   float((pts_arr == 0).mean()),
        # Decay-weighted
        "form_pts_decayed": float((pts_arr * weights).sum() / w_sum),
        "form_gf_decayed":  float((gf_arr * weights).sum() / w_sum),
        "form_ga_decayed":  float((ga_arr * weights).sum() / w_sum),
        "n_recent":         n,
    }


# ── Head-to-head features ─────────────────────────────────────────────────────

def _h2h_features(
    df_past: pd.DataFrame,
    home_team: str,
    away_team: str,
    window: int = 10,
) -> dict[str, float]:
    """
    Head-to-head statistics between home_team and away_team.
    Uses matches from df_past (already filtered to be before current match).
    """
    mask = (
        ((df_past["home_team"] == home_team) & (df_past["away_team"] == away_team)) |
        ((df_past["home_team"] == away_team) & (df_past["away_team"] == home_team))
    )
    h2h = df_past[mask].tail(window)

    if h2h.empty:
        return {
            "h2h_home_win_rate": 0.33,
            "h2h_draw_rate":     0.33,
            "h2h_away_win_rate": 0.33,
            "h2h_home_goals":    1.35,
            "h2h_away_goals":    1.15,
            "h2h_n":             0,
        }

    wins_home = ((h2h["home_team"] == home_team) & (h2h["result"] == "H")).sum() + \
                ((h2h["away_team"] == home_team) & (h2h["result"] == "A")).sum()
    wins_away = ((h2h["home_team"] == away_team) & (h2h["result"] == "H")).sum() + \
                ((h2h["away_team"] == away_team) & (h2h["result"] == "A")).sum()
    draws     = (h2h["result"] == "D").sum()
    n         = len(h2h)

    # Goals from home_team perspective
    hg_as_home = h2h.loc[h2h["home_team"] == home_team, "home_goals"].sum()
    hg_as_away = h2h.loc[h2h["away_team"] == home_team, "away_goals"].sum()
    ag_as_home = h2h.loc[h2h["home_team"] == away_team, "home_goals"].sum()
    ag_as_away = h2h.loc[h2h["away_team"] == away_team, "away_goals"].sum()

    h2h_home_goals = (hg_as_home + hg_as_away) / n
    h2h_away_goals = (ag_as_home + ag_as_away) / n

    return {
        "h2h_home_win_rate": float(wins_home / n),
        "h2h_draw_rate":     float(draws / n),
        "h2h_away_win_rate": float(wins_away / n),
        "h2h_home_goals":    float(h2h_home_goals),
        "h2h_away_goals":    float(h2h_away_goals),
        "h2h_n":             int(n),
    }


# ── League strength normalization ─────────────────────────────────────────────

def _season_strength_stats(df: pd.DataFrame, season: int) -> dict[str, float]:
    """
    Compute league-average attack and defense rates for a given season.
    Used to normalize team-level goals scored/conceded.
    """
    s = df[df["season"] == season]
    if s.empty:
        return {"league_avg_goals": 1.35}
    total_goals = s["home_goals"].sum() + s["away_goals"].sum()
    total_games = len(s) * 2          # each game has 2 'team slots'
    return {"league_avg_goals": float(total_goals / total_games)}


# ── Main entry-point ──────────────────────────────────────────────────────────

def compute_form_features(
    df: pd.DataFrame,
    window: int = 5,
    half_life: float = 60.0,
    h2h_window: int = 10,
) -> pd.DataFrame:
    """
    For every match in `df` (sorted by date), compute form features for
    home and away teams using only matches that occurred BEFORE that match.

    Returns df with additional feature columns prefixed by home_ / away_.
    """
    df = df.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    team_history = _build_team_records(df)

    home_feats_list: list[dict] = []
    away_feats_list: list[dict] = []
    h2h_feats_list:  list[dict] = []

    for idx, row in df.iterrows():
        match_date = row["date"]
        home_team  = row["home_team"]
        away_team  = row["away_team"]

        h_feats = _rolling_features(
            team_history.get(home_team, []), match_date, window, half_life
        )
        a_feats = _rolling_features(
            team_history.get(away_team, []), match_date, window, half_life
        )
        h2h_f = _h2h_features(df.iloc[:idx], home_team, away_team, h2h_window)

        home_feats_list.append({f"home_{k}": v for k, v in h_feats.items()})
        away_feats_list.append({f"away_{k}": v for k, v in a_feats.items()})
        h2h_feats_list.append(h2h_f)

    home_df = pd.DataFrame(home_feats_list, index=df.index)
    away_df = pd.DataFrame(away_feats_list, index=df.index)
    h2h_df  = pd.DataFrame(h2h_feats_list,  index=df.index)

    df = pd.concat([df, home_df, away_df, h2h_df], axis=1)
    return df
