from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data_ingestion.football_data_client import FootballDataClient
from data_ingestion.normalizer import RAW_COLUMNS, normalize_matches


@dataclass
class SyncResult:
    changed: bool
    matches_fetched: int
    rows_written: int
    date_from: date
    date_to: date
    reason: str = ""


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _ensure_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in RAW_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    base_columns = RAW_COLUMNS[:]
    extra_columns = [column for column in df.columns if column not in base_columns]
    return df[base_columns + extra_columns]


def sync_official_matches(cfg: dict) -> SyncResult:
    ingestion_cfg = cfg.get("ingestion", {})
    if not ingestion_cfg.get("enabled", False):
        today = date.today()
        return SyncResult(False, 0, 0, today, today, "Ingestion disabled in config.")

    api_key = os.getenv(ingestion_cfg["api_key_env"], "").strip()
    if not api_key:
        today = date.today()
        return SyncResult(False, 0, 0, today, today, f"Missing env var {ingestion_cfg['api_key_env']}.")

    start_date = date.fromisoformat(ingestion_cfg["start_date"])
    today = date.today()
    state_path = Path(ingestion_cfg["state_path"])
    state = _load_state(state_path)
    lookback_days = int(ingestion_cfg.get("lookback_days", 7))

    last_sync_date = state.get("last_sync_date")
    if last_sync_date:
        date_from = max(start_date, date.fromisoformat(last_sync_date) - timedelta(days=lookback_days))
    else:
        date_from = start_date
    date_to = today

    client = FootballDataClient(api_key=api_key)
    frames = []
    total_fetched = 0
    for competition_code in ingestion_cfg.get("competitions", []):
        matches = client.fetch_finished_matches(competition_code, date_from, date_to)
        total_fetched += len(matches)
        frames.append(normalize_matches(matches, competition_code))

    fresh_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLUMNS)

    raw_path = Path(cfg["data"]["raw_path"])
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path, parse_dates=["date"])
    else:
        raw_df = pd.DataFrame(columns=RAW_COLUMNS)

    raw_df = _ensure_raw_columns(raw_df)

    if fresh_df.empty:
        _save_state(
            state_path,
            {
                "last_sync_date": date_to.isoformat(),
                "last_synced_at": datetime.utcnow().replace(microsecond=0).isoformat(),
                "provider": ingestion_cfg.get("provider"),
                "competitions": ingestion_cfg.get("competitions", []),
            },
        )
        return SyncResult(False, total_fetched, len(raw_df), date_from, date_to, "No finished matches returned.")

    merged_df = pd.concat([raw_df, fresh_df], ignore_index=True, sort=False)
    if "provider_match_id" in merged_df.columns:
        merged_df["provider_match_id"] = merged_df["provider_match_id"].astype("string")
        local_rows = merged_df.loc[merged_df["provider_match_id"].isna()].copy()
        external_rows = merged_df.loc[merged_df["provider_match_id"].notna()].copy()
        if not external_rows.empty:
            external_rows = external_rows.sort_values("last_updated_at").drop_duplicates(
                subset=["provider", "provider_match_id"],
                keep="last",
            )
        merged_df = pd.concat([local_rows, external_rows], ignore_index=True, sort=False)

    merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
    merged_df = merged_df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals", "result"])
    merged_df = merged_df.sort_values(["date", "competition_code", "matchweek", "home_team", "away_team"]).reset_index(drop=True)

    before_rows = len(raw_df.dropna(subset=["date"], how="all")) if len(raw_df) else 0
    after_rows = len(merged_df)
    existing_external = raw_df.loc[raw_df["provider_match_id"].notna()] if "provider_match_id" in raw_df.columns else pd.DataFrame()
    changed = before_rows != after_rows or len(existing_external) != len(merged_df.loc[merged_df["provider_match_id"].notna()])

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(raw_path, index=False)

    _save_state(
        state_path,
        {
            "last_sync_date": date_to.isoformat(),
            "last_synced_at": datetime.utcnow().replace(microsecond=0).isoformat(),
            "provider": ingestion_cfg.get("provider"),
            "competitions": ingestion_cfg.get("competitions", []),
            "rows_in_raw": after_rows,
        },
    )

    return SyncResult(changed, total_fetched, after_rows, date_from, date_to, "")
