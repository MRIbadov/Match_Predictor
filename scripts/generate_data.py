"""
generate_data.py — synthetic Premier League-style dataset generator
"""
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)

TEAMS = [
    "Arsenal","Aston Villa","Brentford","Brighton","Burnley",
    "Chelsea","Crystal Palace","Everton","Fulham","Leeds",
    "Leicester","Liverpool","Man City","Man United","Newcastle",
    "Norwich","Southampton","Tottenham","Watford","West Ham",
]
SEASONS       = [2018,2019,2020,2021,2022,2023,2024,2025]
HOME_ADVANTAGE = 0.25

def generate_team_strengths():
    s={}
    atk_base = rng.normal(0.0,0.3,len(TEAMS))
    dfc_base = rng.normal(0.0,0.3,len(TEAMS))
    for i,team in enumerate(TEAMS):
        s[team]={}
        atk,dfc=atk_base[i],dfc_base[i]
        for season in SEASONS:
            atk+=rng.normal(0.0,0.08); dfc+=rng.normal(0.0,0.08)
            s[team][season]={"attack":float(np.exp(atk)),"defense":float(np.exp(dfc))}
    return s

def simulate_match(home,away,season,strengths):
    mu=1.35
    sh=strengths[home][season]; sa=strengths[away][season]
    lh=np.clip(mu*sh["attack"]*sa["defense"]*(1+HOME_ADVANTAGE),0.2,6.0)
    la=np.clip(mu*sa["attack"]*sh["defense"],0.2,6.0)
    return int(rng.poisson(lh)), int(rng.poisson(la)), lh, la

def main():
    print("Generating team strengths...")
    strengths=generate_team_strengths()
    records=[]
    for season in SEASONS:
        print(f"  Simulating season {season}...")
        fixtures=[(h,a) for h in TEAMS for a in TEAMS if h!=a]
        idx_order=rng.permutation(len(fixtures))
        fixtures=[fixtures[i] for i in idx_order]
        start=datetime.date(season,8,10)
        for idx,(home,away) in enumerate(fixtures):
            mw=(idx//(len(TEAMS)//2))+1
            date=start+datetime.timedelta(weeks=mw-1)
            gh,ga,lh,la=simulate_match(home,away,season,strengths)
            result="H" if gh>ga else ("A" if ga>gh else "D")
            records.append({
                "season":season,"matchweek":mw,"date":date.isoformat(),
                "home_team":home,"away_team":away,
                "home_goals":gh,"away_goals":ga,"result":result,
                "home_shots":max(gh+int(rng.integers(2,8)),1),
                "away_shots":max(ga+int(rng.integers(2,8)),1),
                "home_xg":round(float(max(rng.normal(lh*0.9,0.3),0.1)),2),
                "away_xg":round(float(max(rng.normal(la*0.9,0.3),0.1)),2),
            })
    df=pd.DataFrame(records)
    df["date"]=pd.to_datetime(df["date"])
    df=df.sort_values("date").reset_index(drop=True)
    out=Path("data/raw/matches.csv")
    out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    print(f"\nSaved {len(df):,} matches -> {out}")
    print("Result distribution:\n",df["result"].value_counts(normalize=True).round(3))

if __name__=="__main__":
    main()
