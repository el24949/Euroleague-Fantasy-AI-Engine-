"""Extends prediction/model_v2.py with one more feature: distance_from_norm
-- the player's previous game's deviation from their season-to-date average
going into that previous game. Kept as a separate column from rolling_5/
rolling_3 form so the model can weight recent momentum (this game vs the
player's own norm) independently from plain rolling averages.

Train/test split: restricted to Season 2025 only, then split by Gamecode
within that season (earlier Gamecodes train, later Gamecodes test, no
look-ahead). The previous Gamecode-only split (`df["Gamecode"] < 245`, no
Season filter) silently blended 2023 and 2025 games once both were in the
database -- Gamecode resets to 1 each season, so it compared unrelated
games and put some 2025 games in "train" ahead of earlier 2023 games in
"test", which isn't a valid time-based split. Restricting to one season
first removes the ambiguity entirely.

Reuses add_rolling_form (which now also computes distance_from_norm),
add_rolling_minutes, and add_team_rolling_defense instead of recomputing
any of them.
"""

import sqlite3

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from features.matchup import add_team_rolling_defense
from features.minutes import add_rolling_minutes
from features.rolling_form import add_rolling_form

SEASON = 2025
CUTOFF_GAMECODE = 304  # ~80th percentile game of the 380-game 2025 season


def train_and_predict():
    conn = sqlite3.connect("data/euroleague.db")
    df = pd.read_sql_query(
        "SELECT * FROM boxscores WHERE Season = ? AND Player_ID NOT IN ('Team', 'Total')",
        conn,
        params=[SEASON],
    )
    conn.close()

    df = add_rolling_form(df)
    df = add_rolling_minutes(df)

    team_defense = add_team_rolling_defense(df)
    opp_defense = team_defense.merge(team_defense, on=["Season", "Gamecode"], suffixes=("", "_opp"))
    opp_defense = opp_defense[opp_defense["Team"] != opp_defense["Team_opp"]]
    opp_defense = opp_defense[["Season", "Gamecode", "Team", "rolling_5_conceded_opp"]]
    opp_defense = opp_defense.rename(columns={"rolling_5_conceded_opp": "opp_rolling_5_conceded"})

    df = df.merge(opp_defense, on=["Season", "Gamecode", "Team"], how="left")

    feature_cols = [
        "rolling_5",
        "rolling_3",
        "rolling_5_minutes",
        "rolling_3_minutes",
        "consistency_5",
        "opp_rolling_5_conceded",
        "distance_from_norm",
    ]
    df = df.dropna(subset=feature_cols)

    train = df[df["Gamecode"] < CUTOFF_GAMECODE]
    test = df[df["Gamecode"] >= CUTOFF_GAMECODE]

    # The SQL query already restricts to SEASON, so this should be
    # unreachable -- fail loudly rather than silently training/testing on
    # an unexpected season if that ever stops being true.
    assert (train["Season"] == SEASON).all(), "Train set contains non-2025 rows"
    assert (test["Season"] == SEASON).all(), "Test set contains non-2025 rows"

    X_train, y_train = train[feature_cols], train["base_score"]
    X_test, y_test = test[feature_cols], test["base_score"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    train = train.copy()
    test = test.copy()
    train["predicted_score"] = model.predict(X_train)
    test["predicted_score"] = model.predict(X_test)

    return model, feature_cols, train, test


if __name__ == "__main__":
    model, feature_cols, train, test = train_and_predict()

    test_mae = mean_absolute_error(test["base_score"], test["predicted_score"])

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Gamecode cutoff: {CUTOFF_GAMECODE}")
    print(
        f"Confirmed: train seasons = {sorted(train['Season'].unique())}, "
        f"test seasons = {sorted(test['Season'].unique())} (both 100% Season {SEASON})"
    )
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Coefficients: {dict(zip(feature_cols, model.coef_))}")
    print(f"Intercept: {model.intercept_:.3f}")
