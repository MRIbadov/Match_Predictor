from __future__ import annotations

import json
from datetime import date
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class FootballDataClient:
    """Minimal client for football-data.org v4 competition matches."""

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def fetch_finished_matches(
        self,
        competition_code: str,
        date_from: date,
        date_to: date,
    ) -> list[dict[str, Any]]:
        params = urlencode(
            {
                "dateFrom": date_from.isoformat(),
                "dateTo": date_to.isoformat(),
                "status": "FINISHED",
            }
        )
        url = f"{self.BASE_URL}/competitions/{competition_code}/matches?{params}"
        request = Request(url, headers={"X-Auth-Token": self.api_key})
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("matches", [])
