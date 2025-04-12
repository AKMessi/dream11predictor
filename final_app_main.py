import streamlit as st
import pandas as pd
import joblib
import requests
import os
import gdown
from streamlit_lottie import st_lottie
import json
from streamlit_extras.stylable_container import stylable_container

# Initialize session state keys
for key, default in {
    "access_granted": False,
    "access_key": "",
    "key_validated": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Constants
KEYS_URL = "https://drive.google.com/uc?export=download&id=1iDyBB5lTaXcR5TYLVYc8XAFCHwVwRrJH"

# Fetch keys.json (once only)
def fetch_keys():
    try:
        response = requests.get(KEYS_URL, headers={"Cache-Control": "no-cache"})
        if response.status_code == 200:
            return json.loads(response.content)
        else:
            st.error("⚠️ Failed to fetch access keys.")
            st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading keys: {e}")
        st.stop()

keys_data = fetch_keys()

keys_data = fetch_keys()

# Lock screen if access not granted
if not st.session_state.access_granted:
    with stylable_container(
        key="popup",
        css_styles="""
            {
                border: 2px solid #e50914;
                padding: 20px;
                border-radius: 10px;
                background-color: #fffbe6;
                margin-top: 2em;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            }
        """,
    ):
        st.markdown("### 💰 Choose a Plan")

        plans = [
            {"label": "₹9 for 2 searches", "link": "https://rzp.io/rzp/oT2g6dfL"},
            {"label": "₹27 for 5 searches", "link": "https://rzp.io/rzp/oT2g6dfL"},
            {"label": "₹37 for 10 searches", "link": "https://rzp.io/rzp/oT2g6dfL"},
            {"label": "₹149 for 50 searches", "link": "https://rzp.io/rzp/oT2g6dfL"},
            {"label": "₹247 for 100 searches", "link": "https://rzp.io/rzp/oT2g6dfL"},
        ]

        for plan in plans:
            st.markdown(f"✅ [{plan['label']}]({plan['link']})", unsafe_allow_html=True)

        st.markdown("**🔑 Already have a key?**")
        key_input = st.text_input("Enter your access key", type="password")
        if st.button("🔓 Unlock"):
            st.write("🔍 Keys loaded:", keys_data.keys())

            if key_input in keys_data and keys_data[key_input]["uses_left"] > 0:
                st.session_state.access_granted = True
                st.session_state.key_validated = True
                st.session_state.access_key = key_input
                st.success("✅ Access granted!")
                st.rerun()
            else:
                st.error("❌ Invalid or expired key.")
        st.stop()

# Show UI
st.set_page_config(page_title="Dream11 Predictor", layout="centered")
st_lottie(requests.get("https://assets2.lottiefiles.com/packages/lf20_ydo1amjm.json").json(), height=150)
st.title("🏏 Dream11 Team Predictor")
st.markdown("### 🛠️ Match Setup")

# Show usage counter
current_key = st.session_state.get("access_key")
if current_key and keys_data.get(current_key):
    uses_left = keys_data[current_key].get("uses_left", 0)
    st.markdown(
        f"<div style='text-align:right; font-size:14px; color:green;'>🔄 Uses left: <b>{uses_left}</b></div>",
        unsafe_allow_html=True
    )

# Download necessary files
def download_if_needed(file_id, output):
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

download_if_needed("1dtW9xD3fdoOHrBAFiKX9O18kUUs-BIuw", "final_training_data_with_date.csv")
download_if_needed("1JZn4APJQv2vyRzUSc8Asvqy2T20W2xHK", "player_vs_player_h2h.csv")

# Load model and encoders
model = joblib.load("models/final_model_main.pkl")
encoder = joblib.load("models/role_encoder_main.pkl")
df = pd.read_csv("final_training_data_with_date.csv", encoding="utf-8-sig")
h2h_df = pd.read_csv("player_vs_player_h2h.csv", encoding="utf-8-sig")

# Clean + prep data
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date'], inplace=True)

TEAM_CORRECTIONS = {
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "RCB": "Royal Challengers Bangalore",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings"
}
df.replace({"team": TEAM_CORRECTIONS, "opponent": TEAM_CORRECTIONS}, inplace=True)

def get_most_recent_players(df, team, count=11):
    team_df = df[df['team'] == team]
    if team_df.empty:
        return []
    recent_match = team_df[team_df['date'] == team_df['date'].max()]
    return recent_match['player'].dropna().unique().tolist()[:count]

# Team & Venue setup
teams = sorted(df['team'].dropna().unique())
venues = sorted(df['venue'].dropna().unique())

team1 = st.selectbox("Team 1", teams)
team2 = st.selectbox("Team 2", [t for t in teams if t != team1])
venue = st.selectbox("Venue", venues)

# Player selection
selected_team1, selected_team2 = [], []
recent_team1 = get_most_recent_players(df, team1)
recent_team2 = get_most_recent_players(df, team2)
team1_players = sorted(df[df["team"] == team1]["player"].dropna().unique())
team2_players = sorted(df[df["team"] == team2]["player"].dropna().unique())

st.markdown(f"#### 🟢 {team1} Players")
for i in range(11):
    default = recent_team1[i] if i < len(recent_team1) else None
    selected_team1.append(
        st.selectbox(f"{team1} Player {i+1}", 
                     [p for p in team1_players if p not in selected_team1 or p == default],
                     index=[p for p in team1_players if p not in selected_team1 or p == default].index(default) if default in team1_players else 0,
                     key=f"t1_{i}")
    )

st.markdown(f"#### 🔴 {team2} Players")
for i in range(11):
    default = recent_team2[i] if i < len(recent_team2) else None
    selected_team2.append(
        st.selectbox(f"{team2} Player {i+1}",
                     [p for p in team2_players if p not in selected_team2 or p == default],
                     index=[p for p in team2_players if p not in selected_team2 or p == default].index(default) if default in team2_players else 0,
                     key=f"t2_{i}")
    )

selected_players = selected_team1 + selected_team2

# Prediction button
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

            # 🔄 Update usage count (no rerun)
            if current_key and keys_data.get(current_key):
                keys_data[current_key]["uses_left"] -= 1
                try:
                    requests.post(
                        "https://razorpay-webhook-iub2.onrender.com/update-key",
                        json={
                            "key": current_key,
                            "uses_left": keys_data[current_key]["uses_left"],
                            "secret": "messiisthegoat"
                        },
                        timeout=4
                    )
                except Exception as e:
                    st.warning("🕸️ Usage updated locally, but failed to sync to server.")

            # 🔥 Head-to-Head Matchups
            st.markdown("### 🔥 Head-to-Head Matchups")
            player_team_map = {p: team1 for p in selected_team1} | {p: team2 for p in selected_team2}

            h2h_matchups = h2h_df[
                (h2h_df["batter"].isin(selected_players)) &
                (h2h_df["bowler"].isin(selected_players)) &
                (h2h_df["batter"] != h2h_df["bowler"])
            ]

            insights = []
            for _, row in h2h_matchups.iterrows():
                batter = row["batter"]
                bowler = row["bowler"]

                # only show if from opposing teams
                if player_team_map.get(batter) != player_team_map.get(bowler):
                    dismissals = int(row["dismissals"])
                    avg = row.get("average_vs_bowler", 0.0)
                    sr = row.get("strike_rate", 0.0)
                    balls = row.get("balls_faced", 0)

                    if dismissals >= 2:
                        insights.append(f"🛑 **{bowler}** dismissed **{batter}** `{dismissals}` times.")
                    elif avg >= 35 and balls >= 10:
                        insights.append(f"🚀 **{batter}** averages `{avg:.1f}` (SR `{sr:.1f}`) vs **{bowler}**.")

            if insights:
                for tip in insights:
                    st.markdown(f"- {tip}")
            else:
                st.info("ℹ️ No notable head-to-head matchups found between opposing players.")

            # 💡 Fantasy Tips
            st.markdown("### 💡 Fantasy Tips")
            st.markdown("""
            🧠 **Use this team as a base only** – don’t copy it blindly. Blend it with your own instincts.\n
            🟨 Wait for toss + confirmed XI before finalizing.\n
            🚫 This is NOT financial advice. Play responsibly.
            """)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

