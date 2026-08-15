"""Same three features and time-based split as prediction/model_matchup.py,
but with a RandomForestRegressor instead of LinearRegression.

An unconstrained forest (n_estimators=100, everything else default) badly
overfit: train MAE 2.07 vs test MAE 5.69. This version regularizes it with
a small manual grid over max_depth and min_samples_leaf, to see whether
constraining tree growth closes that gap and can beat the linear model's
test MAE of 5.275.

Reuses add_rolling_form (features/rolling_form.py), add_rolling_minutes
(features/minutes.py), and add_team_rolling_defense (features/matchup.py)
instead of recomputing any of them. Train/test split: restricted to Season
2025 only, then split by Gamecode within that season -- see
prediction/model.py's docstring for why the Season filter is required
(Gamecode resets each season, so a Gamecode-only split silently blended
2023 and 2025 games once both were in the database).
"""

import sqlite3

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

    feature_cols = ["rolling_5", "rolling_5_minutes", "opp_rolling_5_conceded"]
    df = df.dropna(subset=feature_cols)

    train = df[df["Gamecode"] < CUTOFF_GAMECODE]
    test = df[df["Gamecode"] >= CUTOFF_GAMECODE]

    assert (train["Season"] == SEASON).all(), "Train set contains non-2025 rows"
    assert (test["Season"] == SEASON).all(), "Test set contains non-2025 rows"

    X_train, y_train = train[feature_cols], train["base_score"]
    X_test, y_test = test[feature_cols], test["base_score"]

    LINEAR_BASELINE_MAE = 5.275

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Gamecode cutoff: {CUTOFF_GAMECODE}")
    print(
        f"Confirmed: train seasons = {sorted(train['Season'].unique())}, "
        f"test seasons = {sorted(test['Season'].unique())} (both 100% Season {SEASON})"
    )
    print(
        f"{'max_depth':>10} {'min_samples_leaf':>18} {'train_mae':>10} "
        f"{'test_mae':>10} {'gap':>8} {'beats_linear':>13}"
    )

    for max_depth in (4, 5, 6):
        for min_samples_leaf in (20, 35, 50):
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
            model.fit(X_train, y_train)

            train_mae = mean_absolute_error(y_train, model.predict(X_train))
            test_mae = mean_absolute_error(y_test, model.predict(X_test))
            gap = test_mae - train_mae
            beats_linear = "yes" if test_mae < LINEAR_BASELINE_MAE else "no"

            print(
                f"{max_depth:>10} {min_samples_leaf:>18} {train_mae:>10.3f} "
                f"{test_mae:>10.3f} {gap:>8.3f} {beats_linear:>13}"
            )
