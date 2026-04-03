import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="🏏 IPL Analytics Pro", layout="wide")

# =========================
# LOAD DATA (HYBRID)
# =========================
@st.cache_data
def load_data():
    import pandas as pd
    import os
    import zipfile

    # ───────────────
    # 1. TRY KAGGLE
    # ───────────────
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "chaitu20/ipl-dataset2008-2025",
            "IPL.csv"
        )

        print("✅ Loaded from Kaggle")
        return df

    except Exception as e:
        print("⚠️ Kaggle failed:", e)

    # ───────────────
    # 2. TRY LOCAL CSV
    # ───────────────
    try:
        if os.path.exists("IPL.csv"):
            print("✅ Loaded from IPL.csv")
            return pd.read_csv("IPL.csv")
    except Exception as e:
        print("⚠️ CSV load failed:", e)

    # ───────────────
    # 3. TRY ZIP FILE
    # ───────────────
    try:
        # check current folder
        for file in os.listdir():
            if file.endswith(".zip"):
                print(f"✅ Found ZIP: {file}")

                with zipfile.ZipFile(file, 'r') as z:
                    for name in z.namelist():
                        if name.endswith(".csv"):
                            print(f"✅ Loading {name} from ZIP")
                            return pd.read_csv(z.open(name))

        # check inside subfolders
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    print(f"✅ Found ZIP in folder: {zip_path}")

                    with zipfile.ZipFile(zip_path, 'r') as z:
                        for name in z.namelist():
                            if name.endswith(".csv"):
                                print(f"✅ Loading {name} from ZIP")
                                return pd.read_csv(z.open(name))

    except Exception as e:
        print("⚠️ ZIP load failed:", e)

    # ───────────────
    # ❌ FINAL ERROR
    # ───────────────
    raise Exception("🚨 No dataset found (Kaggle / CSV / ZIP all failed)")

df = load_data()

# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    df = df.copy()

    # ── Standardise column names ──
    df.columns = df.columns.str.lower().str.strip()

    # ── Fix column names ──
    rename_map = {
        "runs_batter": "batsman_runs",
        "runs_total": "total_runs",
        "player_out": "player_dismissed",
        "wicket_kind": "dismissal_kind",
        "valid_ball": "is_legal",
    }

    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # ── Ensure batsman column exists ──
    if "batter" in df.columns:
        df.rename(columns={"batter": "batsman"}, inplace=True)

    # ── Create is_wicket column ──
    if "player_dismissed" in df.columns:
        df["is_wicket"] = df["player_dismissed"].notna().astype(int)
    else:
        df["is_wicket"] = 0

    # ── FIX: Year conversion (IMPORTANT for ML) ──
    if "season" in df.columns:
        df["year"] = (
            df["season"]
            .astype(str)
            .str.extract(r"(\d{4})")[0]   # extract first year
        )
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

    # Convert year to numeric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Drop invalid year rows
    df = df.dropna(subset=["year"])

    # ── Numeric conversions ──
    for col in ["batsman_runs", "total_runs", "over", "ball"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Fill missing values safely ──
    df["batsman_runs"] = df.get("batsman_runs", 0).fillna(0)
    df["total_runs"] = df.get("total_runs", 0).fillna(0)
    df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)

    # ── Derived features ──
    df["is_four"] = (df["batsman_runs"] == 4).astype(int)
    df["is_six"] = (df["batsman_runs"] == 6).astype(int)

    # ── Ensure is_legal exists ──
    if "is_legal" not in df.columns:
        df["is_legal"] = 1

    return df

df = clean_data(df)

# =========================
# IPL 2026 TEAMS
# =========================
ipl_teams = [
    "Chennai Super Kings","Mumbai Indians","Royal Challengers Bangalore",
    "Kolkata Knight Riders","Delhi Capitals","Punjab Kings",
    "Rajasthan Royals","Sunrisers Hyderabad",
    "Lucknow Super Giants","Gujarat Titans"
]

df = df[df['batting_team'].isin(ipl_teams)]

# =========================
# UI HEADER
# =========================
st.title("🏏 IPL Analytics Pro Dashboard")
st.markdown("### 📊 Advanced Analytics + Predictions (2026)")

# =========================
# KPI
# =========================
c1,c2,c3,c4 = st.columns(4)

c1.metric("Runs", df['batsman_runs'].sum())
c2.metric("Wickets", df['is_wicket'].sum())
c3.metric("Matches", df['match_id'].nunique())
c4.metric("Players", df['batsman'].nunique())

# =========================
# TOP BATSMEN (15)
# =========================
st.subheader("🔥 Top 15 Batsmen")

top_bat = df.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(15).reset_index()

fig = px.bar(
    top_bat,
    x='batsman',
    y='batsman_runs',
    color='batsman_runs',
    color_continuous_scale='oranges',
    title="Top 15 Run Scorers"
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# TOP BOWLERS (15)
# =========================
st.subheader("🎯 Top 15 Bowlers")

top_bowl = (
    df[df['is_wicket']==1]
    .groupby('bowler')
    .size()
    .sort_values(ascending=False)
    .head(15)
    .reset_index(name='wickets')
)

fig = px.bar(
    top_bowl,
    x='bowler',
    y='wickets',
    color='wickets',
    color_continuous_scale='reds',
    title="Top 15 Wicket Takers"
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# TEAM PIE CHART
# =========================
st.subheader("🥧 Team Run Contribution")

team_runs = df.groupby('batting_team')['batsman_runs'].sum().reset_index()

fig = px.pie(
    team_runs,
    names='batting_team',
    values='batsman_runs',
    color_discrete_sequence=px.colors.sequential.Rainbow
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# ORANGE CAP (FILTERED)
# =========================
st.subheader("🧡 Orange Cap Prediction (2026)")

player_year = df.groupby(['year','batsman'])['batsman_runs'].sum().reset_index()
top_players = player_year.groupby('batsman')['batsman_runs'].sum().nlargest(15).index

pred = []
for p in top_players:
    pdata = player_year[player_year['batsman']==p]
    if len(pdata)>2:
        model = LinearRegression().fit(pdata[['year']], pdata['batsman_runs'])
        pred.append((p, model.predict([[2026]])[0]))

orange_df = pd.DataFrame(pred, columns=['Player','Runs'])
orange_df['%'] = (orange_df['Runs']/orange_df['Runs'].sum()*100).round(2)
orange_df = orange_df.sort_values('%', ascending=False).reset_index(drop=True)
orange_df.index += 1

st.dataframe(orange_df)

st.success(f"🏆 Orange Cap Winner 2026: {orange_df.iloc[0]['Player']}")

# =========================
# PURPLE CAP (FILTERED)
# =========================
st.subheader("💜 Purple Cap Prediction (2026)")

wk = df[df['is_wicket']==1].groupby(['year','bowler']).size().reset_index(name='w')
top_bowlers = wk.groupby('bowler')['w'].sum().nlargest(15).index

pred=[]
for b in top_bowlers:
    bdata = wk[wk['bowler']==b]
    if len(bdata)>2:
        model = LinearRegression().fit(bdata[['year']], bdata['w'])
        pred.append((b, model.predict([[2026]])[0]))

purple_df = pd.DataFrame(pred, columns=['Bowler','Wickets'])
purple_df['%'] = (purple_df['Wickets']/purple_df['Wickets'].sum()*100).round(2)
purple_df = purple_df.sort_values('%', ascending=False).reset_index(drop=True)
purple_df.index += 1

st.dataframe(purple_df)

st.success(f"🏆 Purple Cap Winner 2026: {purple_df.iloc[0]['Bowler']}")

# =========================
# IPL WINNER
# =========================
st.subheader("🏆 IPL 2026 Winner Prediction")

team_year = df.groupby(['year','batting_team'])['batsman_runs'].sum().reset_index()

pred=[]
for t in ipl_teams:
    tdata = team_year[team_year['batting_team']==t]
    if len(tdata)>2:
        model = LinearRegression().fit(tdata[['year']], tdata['batsman_runs'])
        pred.append((t, model.predict([[2026]])[0]))

team_df = pd.DataFrame(pred, columns=['Team','Score'])
team_df['Win %'] = (team_df['Score']/team_df['Score'].sum()*100).round(2)
team_df = team_df.sort_values('Win %', ascending=False).reset_index(drop=True)
team_df.index += 1

team_df = team_df[['Team','Win %']]

st.dataframe(team_df)

st.success(f"🏆 IPL 2026 Winner: {team_df.iloc[0]['Team']} ({team_df.iloc[0]['Win %']}%)")

st.markdown("---")
st.markdown("Made using Streamlit & Plotly by Chirag Beuria | Data Source: Kaggle (IPL Dataset 2008-2025) ")