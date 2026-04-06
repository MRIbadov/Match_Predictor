"""
FastAPI backend and frontend entrypoint for the football predictor app.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import Predictor


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
INDEX_PATH = FRONTEND_DIR / "index.html"

with open(BASE_DIR / "config" / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(
    title="Football Match Predictor",
    description="ML-powered football match outcome and scoreline predictor.",
    version=cfg["project"]["version"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


class PredictRequest(BaseModel):
    home_team: str
    away_team: str

    @field_validator("home_team", "away_team")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        return value.strip().title()


class ScorelineProbability(BaseModel):
    scoreline: str
    probability: float


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    expected_home_goals: float
    expected_away_goals: float
    top_scorelines: list[ScorelineProbability]
    model_version: str


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": cfg["project"]["version"]}


@app.get("/teams")
def list_teams() -> dict[str, list[str]]:
    pred = get_predictor()
    return {"teams": pred.available_teams}


@app.post("/predict", response_model=PredictResponse)
def predict_post(req: PredictRequest) -> PredictResponse:
    return _run_prediction(req.home_team, req.away_team)


@app.get("/predict/{home}/{away}", response_model=PredictResponse)
def predict_get(home: str, away: str) -> PredictResponse:
    return _run_prediction(home.title(), away.title())


def _run_prediction(home_team: str, away_team: str) -> PredictResponse:
    pred = get_predictor()

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="Home and away team must be different.")

    unknown = [team for team in [home_team, away_team] if team not in pred.available_teams]
    if unknown:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team(s): {unknown}. Use GET /teams for valid names.",
        )

    result = pred.predict(home_team, away_team)
    return PredictResponse(**result, model_version=cfg["project"]["version"])
