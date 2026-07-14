"""Naive fantasy-score prediction: a player's rolling_5 form going into a
game, compared against what they actually scored that game.

Reuses add_rolling_form from features/rolling_form.py (which itself reuses
base_fantasy_score from scoring/base_score.py), so the rolling window logic
and the score formula each stay in exactly one place.
"""

import sqlite3

import pandas as pd

from features.rolling_form import add_rolling_form


def get_predictions():
    conn = sqlite3.connect("data/euroleague.db")
    df = pd.read_sql_query(
        "SELECT * FROM boxscores WHERE Player_ID NOT IN ('Team', 'Total')",
        conn,
    )
    conn.close()

    df = add_rolling_form(df)

    by_player = df.groupby("Player_ID")["base_score"]
    df["prior_games_count"] = by_player.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).count()
    )

    df = df.dropna(subset=["rolling_5"])

    df["predicted_score"] = df["rolling_5"]
    df["actual_score"] = df["base_score"]

    return df[
        [
            "Player",
            "Season",
            "Gamecode",
            "predicted_score",
            "actual_score",
            "prior_games_count",
        ]
    ]


if __name__ == "__main__":
    result = get_predictions()
    columns = ["Player", "Season", "Gamecode", "predicted_score", "actual_score"]
    print(result[columns].head(15).to_string(index=False))
