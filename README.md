# Football Predictor

## 1. Overview Of Project

This project predicts football match outcomes using machine learning. It estimates:

- home win probability
- draw probability
- away win probability
- expected goals
- likely scorelines

It also includes a FastAPI backend so predictions can be served as an application, not only as a training script.

## 2. What Tools Have Been Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SciPy
- FastAPI
- Uvicorn
- YAML configuration

## 3. What Was The Goal

The goal was to build an end-to-end football prediction system that:

- creates or loads match data
- engineers useful football features
- trains ML models
- evaluates performance on future seasons
- serves predictions through an API

## 4. How I Achieved It

I achieved this by building a full pipeline with these steps:

1. Generated match data for seasons `2018-2025`
2. Created features such as Elo ratings, rolling form, venue stats, and strength metrics
3. Trained two models:
   - Poisson goal model
   - XGBoost classifier
4. Combined both models with an ensemble
5. Evaluated the system using time-based train, validation, and test splits
6. Exposed predictions through FastAPI endpoints

## 5. Potential Improvements

- Add official real match data ingestion with `football-data.org`
- Support more leagues and competitions
- Improve draw prediction
- Add automated retraining

## Short Summary

This project is a complete football ML pipeline that goes from data to prediction API. It was built to show both machine learning workflow and practical software engineering in one system.
