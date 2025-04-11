import streamlit as st
import pandas as pd
import joblib
import requests
import os
import gdown
from streamlit_lottie import st_lottie
import json
import os

# Load keys
KEYS_PATH = "keys.json"

if not os.path.exists(KEYS_PATH):
    with open(KEYS_PATH, "w") as f:
        json.dump({}, f)

with open(KEYS_PATH, "r") as f:
    keys_data = json.load(f)

# Ask for key if not authenticated
if "key_validated" not in st.session_state or not st.session_state.key_validated:
    st.warning("🔒 Please enter your access key to continue.")
    key_input = st.text_input("Access Key", type="password")

    if st.button("🔓 Unlock"):
        if key_input in keys_data and keys_data[key_input]["uses_left"] > 0:
            st.session_state.key_validated = True
            st.session_state.access_key = key_input
            st.success("✅ Access granted!")
            st.rerun()
        else:
            st.error("❌ Invalid or expired key.")
    st.stop()


# Lottie animation loader
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

# UI config
st.set_page_config(page_title="Dream11 Predictor", layout="centered")
st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_ydo1amjm.json"), height=150)

st.title("🏏 Dream11 Team Predictor")
st.markdown("### 🛠️ Match Setup")

# Ensure CSVs are downloaded
def download_if_needed(file_id, output):
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

download_if_needed("1dtW9xD3fdoOHrBAFiKX9O18kUUs-BIuw", "final_training_data_with_date.csv")
download_if_needed("1JZn4APJQv2vyRzUSc8Asvqy2T20W2xHK", "player_vs_player_h2h.csv")

# Load model and encoders
model = joblib.load("models/final_model_main.pkl")
encoder = joblib.load("models/role_encoder_main.pkl")

# Load datasets
df = pd.read_csv("final_training_data_with_date.csv", encoding="utf-8-sig")

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Standardize team names
TEAM_CORRECTIONS = {
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "RCB": "Royal Challengers Bangalore",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings"
    # Add any more aliases here
}
df['team'] = df['team'].replace(TEAM_CORRECTIONS)
df['opponent'] = df['opponent'].replace(TEAM_CORRECTIONS)

h2h_df = pd.read_csv("player_vs_player_h2h.csv", encoding="utf-8-sig")

def get_most_recent_players(df, team_name, count=11):
    team_df = df[df['team'] == team_name]
    if team_df.empty:
        return []

    latest_date = team_df['date'].max()
    recent_match = team_df[team_df['date'] == latest_date]
    return recent_match['player'].dropna().unique().tolist()[:count]



# Team and Venue Selection
teams = sorted(df['team'].dropna().unique())
venues = sorted(df['venue'].dropna().unique())

team1 = st.selectbox("Team 1", teams)
team2 = st.selectbox("Team 2", [t for t in teams if t != team1])
venue = st.selectbox("Venue", venues)

# Select Players
st.markdown(f"#### 🟢 {team1} Players")
selected_team1 = []
team1_players = sorted(df[df["team"] == team1]["player"].dropna().unique())
recent_team1 = get_most_recent_players(df, team1)

for i in range(11):
    default_player = recent_team1[i] if i < len(recent_team1) else None
    selected_team1.append(
        st.selectbox(f"{team1} Player {i+1}", 
                     [p for p in team1_players if p not in selected_team1 or p == default_player],
                     index=[p for p in team1_players if p not in selected_team1 or p == default_player].index(default_player) if default_player in team1_players else 0,
                     key=f"t1_{i}")
    )


st.markdown(f"#### 🔴 {team2} Players")
selected_team2 = []
team2_players = sorted(df[df["team"] == team2]["player"].dropna().unique())
recent_team2 = get_most_recent_players(df, team2)

for i in range(11):
    default_player = recent_team2[i] if i < len(recent_team2) else None
    selected_team2.append(
        st.selectbox(f"{team2} Player {i+1}", 
                     [p for p in team2_players if p not in selected_team2 or p == default_player],
                     index=[p for p in team2_players if p not in selected_team2 or p == default_player].index(default_player) if default_player in team2_players else 0,
                     key=f"t2_{i}")
    )


selected_players = selected_team1 + selected_team2

# Access key management
with open("keys.json", "r") as f:
    keys_data = json.load(f)

if "access_granted" not in st.session_state:
    st.session_state.access_granted = False
    st.session_state.key_used = None

if not st.session_state.access_granted:
    key_input = st.text_input("🔑 Enter Access Key")
    if st.button("Unlock"):
        if key_input in keys_data:
            if keys_data[key_input]["uses_left"] > 0:
                st.session_state.access_granted = True
                st.session_state.key_used = key_input
                st.success("✅ Access granted!")
            else:
                st.error("❌ No searches left. Please purchase more.")
        else:
            st.error("❌ Invalid access key.")
    st.stop()


# Predict
if st.button("🔮 Predict Best XI"):

    match_data = df[df["player"].isin(selected_players) & (df["venue"] == venue)].copy()

    if match_data.empty:
        st.error("❌ No data for selected players at this venue.")
    else:
        try:
            match_data[["team", "opponent", "role", "venue"]] = encoder.transform(
                match_data[["team", "opponent", "role", "venue"]]
            )
        except Exception as e:
            st.warning(f"⚠️ Encoding failed: {e}")

        feature_cols = [
            "team", "opponent", "role", "venue", "avg_recent_points", "consistency",
            "avg_points_vs_role", "avg_h2h_points", "venue_points", "avg_fantasy_points_vs_bowlers",
            "avg_h2h_fantasy", "avg_h2h_sr", "avg_h2h_avg", "h2h_avg_fantasy_points_vs_bowlers",
            "h2h_avg_strike_rate_vs_bowlers", "h2h_avg_avg_vs_bowlers", "venue_batting_matches",
            "venue_batting_innings", "venue_total_runs", "venue_batting_dismissals", "venue_batting_average",
            "venue_bowling_matches", "balls_bowled_y", "wickets_y", "runs_conceded_y", "economy_y", "bowling_average_y"
        ]

        match_data = match_data.loc[:, ~match_data.columns.duplicated()]
        for col in feature_cols:
            match_data[col] = pd.to_numeric(match_data[col], errors="coerce").fillna(0)

        try:
            match_data["predicted_score"] = model.predict(match_data[feature_cols])
            match_data = match_data.drop_duplicates("player").sort_values("predicted_score", ascending=False).reset_index(drop=True)
            final_team = match_data.head(11).copy()
            final_team["designation"] = "-"
            if not final_team.empty:
                final_team.at[0, "designation"] = "C"
                final_team.at[1, "designation"] = "VC"

            st.success("✅ Predicted Best XI")
            st.dataframe(final_team[["player", "team", "role", "predicted_score", "designation"]])

            # ✅ Only deduct usage if prediction succeeded
            current_key = st.session_state.get("access_key")
            if current_key and keys_data.get(current_key):
                keys_data[current_key]["uses_left"] -= 1
                with open(KEYS_PATH, "w") as f:
                    json.dump(keys_data, f, indent=2)

            # Head-to-Head Matchups
            st.markdown("### 🔥 Head-to-Head Matchups")
            player_team_map = {p: team1 for p in selected_team1}
            player_team_map.update({p: team2 for p in selected_team2})

            h2h_matchups = h2h_df[
                (h2h_df["batter"].isin(selected_players)) &
                (h2h_df["bowler"].isin(selected_players)) &
                (h2h_df["batter"] != h2h_df["bowler"])
            ]

            insights = []
            for _, row in h2h_matchups.iterrows():
                batter, bowler = row["batter"], row["bowler"]
                if player_team_map.get(batter) != player_team_map.get(bowler):
                    if row["dismissals"] >= 2:
                        insights.append(f"🛑 **{bowler}** dismissed **{batter}** {int(row['dismissals'])} times")
                    elif row["average_vs_bowler"] >= 35 and row["balls_faced"] >= 10:
                        insights.append(f"🚀 **{batter}** averages {row['average_vs_bowler']:.1f} (SR {row['strike_rate']:.1f}) vs **{bowler}**")

            if insights:
                for tip in insights:
                    st.markdown(f"- {tip}")
            else:
                st.info("No notable head-to-head matchups found.")

            # Final Tips
            st.markdown("### 💡 Fantasy Tips")
            st.markdown("""
            🧠 **Use this team as a base only** – don't copy it blindly. Blend it with your own instincts.\n
            🟨 Wait for toss + confirmed XI before finalizing.\n
            🚫 This is NOT financial advice. Play responsibly.
            """)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
