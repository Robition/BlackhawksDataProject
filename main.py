import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

ACCESS_LEVEL = "trial"
LANGUAGE = "en"
FORMAT = "json"

GAME_IDS = [
    "2c79c49d-c247-42c4-9126-a65c0ec1a1be",
    "2010fc1d-05a9-46a9-ab51-57941d7e31a7",
]

def fetch_pbp(game_id):
    url = f"https://api.sportradar.com/nhl/{ACCESS_LEVEL}/v7/{LANGUAGE}/games/{game_id}/pbp.{FORMAT}?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to get PBP for game {game_id}: {response.status_code}")
        return None
    return response.json()

def extract_shots_from_pbp(pbp_json):
    shots = []
    if "events" not in pbp_json:
        return shots

    for event in pbp_json["events"]:
        event_type = event.get("type")
        if event_type in ["shot", "goal"]:
            location = event.get("location") or {}
            x = location.get("x")
            y = location.get("y")
            shot_type = event.get("shot_type", "unknown")
            is_goal = 1 if event_type == "goal" else 0
            if x is not None and y is not None:
                shots.append({
                    "x_location": x,
                    "y_location": y,
                    "play_type": shot_type,
                    "goal": is_goal
                })
    return shots

def main():
    all_shots = []

    for game_id in GAME_IDS:
        print(f"Fetching PBP data for game {game_id}")
        pbp_data = fetch_pbp(game_id)
        if pbp_data:
            shots = extract_shots_from_pbp(pbp_data)
            all_shots.extend(shots)

    if not all_shots:
        print("No shot data found.")
        return

    df = pd.DataFrame(all_shots)
    print(f"Total shots collected: {len(df)}")

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    play_type_encoded = encoder.fit_transform(df[["play_type"]])

    X = np.hstack((df[["x_location", "y_location"]].values, play_type_encoded))
    y = df["goal"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
