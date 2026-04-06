from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


RAW_COLUMNS = [
    "provider",
    "provider_match_id",
    "competition_code",
    "season",
    "matchweek",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_shots",
    "away_shots",
    "home_xg",
    "away_xg",
    "last_updated_at",
]


def normalize_match(match: dict[str, Any], competition_code: str) -> dict[str, Any] | None:
    score = match.get("score", {}).get("fullTime", {})
    home_goals = score.get("home")
    away_goals = score.get("away")

    if home_goals is None or away_goals is None:
        return None

    utc_date = match.get("utcDate")
    if not utc_date:
        return None

    kick_off = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
    season_start = match.get("season", {}).get("startDate", "")
    season = int(season_start[:4]) if season_start else kick_off.year

    if home_goals > away_goals:
        result = "H"
    elif away_goals > home_goals:
        result = "A"
    else:
        result = "D"

    return {
        "provider": "football-data.org",
        "provider_match_id": str(match["id"]),
        "competition_code": competition_code,
        "season": season,
        "matchweek": match.get("matchday"),
        "date": kick_off.date().isoformat(),
        "home_team": match.get("homeTeam", {}).get("name"),
        "away_team": match.get("awayTeam", {}).get("name"),
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "result": result,
        "home_shots": pd.NA,
        "away_shots": pd.NA,
        "home_xg": pd.NA,
        "away_xg": pd.NA,
        "last_updated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
    }


def normalize_matches(matches: list[dict[str, Any]], competition_code: str) -> pd.DataFrame:
    rows = [row for row in (normalize_match(match, competition_code) for match in matches) if row is not None]
    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)
