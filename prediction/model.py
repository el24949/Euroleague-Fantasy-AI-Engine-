"""First real regression model: predict a player's fantasy score from their
leakage-free rolling form and rolling minutes.

Reuses add_rolling_form (features/rolling_form.py) and add_rolling_minutes
(features/minutes.py) instead of recomputing either. Train/test split is by
time (earlier Gamecodes train, later Gamecodes test), never random -- a
random split would leak future games into training and make the test score
look better than the model would actually do in production.
"""

import sqlite3

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from features.minutes import add_rolling_minutes
from features.rolling_form import add_rolling_form

CUTOFF_GAMECODE = 245  # ~80th percentile game of the 306-game season

if __name__ == "__main__":
    conn = sqlite3.connect("data/euroleague.db")
    df = pd.read_sql_query(
        "SELECT * FROM boxscores WHERE Player_ID NOT IN ('Team', 'Total')",
        conn,
    )
    conn.close()

    df = add_rolling_form(df)
    df = add_rolling_minutes(df)

    feature_cols = ["rolling_5", "rolling_5_minutes"]
    df = df.dropna(subset=feature_cols)

    train = df[df["Gamecode"] < CUTOFF_GAMECODE]
    test = df[df["Gamecode"] >= CUTOFF_GAMECODE]

    X_train, y_train = train[feature_cols], train["base_score"]
    X_test, y_test = test[feature_cols], test["base_score"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Test MAE: {mae:.3f}")
    print(f"Coefficients: {dict(zip(feature_cols, model.coef_))}")
    print(f"Intercept: {model.intercept_:.3f}")
