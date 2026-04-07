import pandas as pd
import pickle
from xgboost import XGBClassifier

# Dummy training data (temporary)
data = {
    'team1': [0,1,2,3,4],
    'team2': [1,2,3,4,0],
    'venue': [0,1,2,3,4],
    'toss_winner': [0,1,2,3,4],
    'toss_decision': [0,1,0,1,0],
    'team1_form': [3,4,2,5,1],
    'team2_form': [4,3,5,2,1],
    'team1_batting': [2800,3000,2500,3200,2700],
    'team2_batting': [3300,2900,3100,2800,2600],
    'team1_bowling': [90,100,85,95,88],
    'team2_bowling': [130,120,110,115,105],
    'dew': [1,0,1,0,1],
    'is_night': [1,1,0,1,0],
    'winner': [0,1,2,3,4]
}

df = pd.DataFrame(data)

X = df.drop('winner', axis=1)
y = df['winner']

model = XGBClassifier()
model.fit(X, y)

# Save trained model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved!")