import streamlit as st
import pandas as pd
import pickle

# Load model only
model = pickle.load(open('model.pkl', 'rb'))

st.title("🏏 IPL Match Prediction System")

# Team & venue lists
teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
         'Rajasthan Royals', 'Sunrisers Hyderabad', 'Lucknow Super Giants',
         'Gujarat Titans']

venues = ['Wankhede Stadium', 'Eden Gardens', 'M Chinnaswamy Stadium',
          'Feroz Shah Kotla', 'MA Chidambaram Stadium',
          'Rajiv Gandhi International Stadium',
          'Punjab Cricket Association Stadium',
          'Sawai Mansingh Stadium', 'Narendra Modi Stadium',
          'Barsapara Cricket Stadium']# Manual mappings (IMPORTANT)
team_mapping = {team: i for i, team in enumerate(teams)}
venue_mapping = {venue: i for i, venue in enumerate(venues)}

# Reverse mapping for prediction output
reverse_team_mapping = {i: team for team, i in team_mapping.items()}

# User inputs
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)
venue = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

# Default features (can improve later)
team1_form = 3
team2_form = 4
team1_batting = 2800
team2_batting = 3300
team1_bowling = 90
team2_bowling = 130
dew = 1
is_night = 1

if st.button("Predict Winner"):

    sample = pd.DataFrame([{
        'team1': team_mapping[team1],
        'team2': team_mapping[team2],
        'venue': venue_mapping[venue],
        'toss_winner': team_mapping[toss_winner],
        'toss_decision': 1 if toss_decision == "field" else 0,
        'team1_form': team1_form,
        'team2_form': team2_form,
        'team1_batting': team1_batting,
        'team2_batting': team2_batting,
        'team1_bowling': team1_bowling,
        'team2_bowling': team2_bowling,
        'dew': dew,
        'is_night': is_night
    }])

    # Convert to float
    sample = sample.astype(float)

    # Prediction
    prediction = model.predict(sample.values)
    prob = model.predict_proba(sample.values)

    winner = reverse_team_mapping[int(prediction[0])]

    st.success(f"🏆 Predicted Winner: {winner}")
    st.write(f"📊 Winning Probability: {round(max(prob[0])*100,2)}%")