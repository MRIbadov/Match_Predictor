from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.sync import sync_official_matches


def main() -> None:
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = sync_official_matches(cfg)
    print("Official data sync")
    print(f"  Window: {result.date_from} -> {result.date_to}")
    print(f"  Matches fetched: {result.matches_fetched}")
    print(f"  Rows now in raw data: {result.rows_written}")
    print(f"  Changed: {result.changed}")
    if result.reason:
        print(f"  Note: {result.reason}")


if __name__ == "__main__":
    main()
