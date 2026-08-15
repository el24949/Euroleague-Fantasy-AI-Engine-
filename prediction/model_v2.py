"""LinearRegression with an expanded feature set: rolling_5 and rolling_3
form, rolling_5 and rolling_3 minutes, consistency (rolling_5 std of
fantasy score), and opponent rolling_5 defensive rating.

Train/test split: restricted to Season 2025 only, then split by Gamecode
within that season (earlier Gamecodes train, later Gamecodes test, no
look-ahead) -- see prediction/model.py's docstring for why the Season
filter is required (Gamecode resets each season, so a Gamecode-only split
silently blended 2023 and 2025 games once both were in the database).
Reuses add_rolling_form (which now also computes consistency_5),
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

if __name__ == "__main__":
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
    ]
    df = df.dropna(subset=feature_cols)

    train = df[df["Gamecode"] < CUTOFF_GAMECODE]
    test = df[df["Gamecode"] >= CUTOFF_GAMECODE]

    assert (train["Season"] == SEASON).all(), "Train set contains non-2025 rows"
    assert (test["Season"] == SEASON).all(), "Test set contains non-2025 rows"

    X_train, y_train = train[feature_cols], train["base_score"]
    X_test, y_test = test[feature_cols], test["base_score"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Gamecode cutoff: {CUTOFF_GAMECODE}")
    print(
        f"Confirmed: train seasons = {sorted(train['Season'].unique())}, "
        f"test seasons = {sorted(test['Season'].unique())} (both 100% Season {SEASON})"
    )
    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Coefficients: {dict(zip(feature_cols, model.coef_))}")
    print(f"Intercept: {model.intercept_:.3f}")
