"""Per-player rolling fantasy-score features: recent form trend.

Reuses the single source-of-truth base_fantasy_score from scoring/base_score.py
so the score formula is never duplicated here. Rolling averages are shifted
so a game's own score is never included in its own rolling window (no data
leakage) -- only strictly earlier games in that player's chronological
history are used, matching the approach in features/minutes.py.
"""

import sqlite3

import pandas as pd

from scoring.base_score import base_fantasy_score


def add_rolling_form(df):
    df["base_score"] = df.apply(base_fantasy_score, axis=1)

    df = df.sort_values(["Player_ID", "Season", "Gamecode"]).reset_index(drop=True)

    by_player = df.groupby("Player_ID")["base_score"]
    df["rolling_3"] = by_player.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling_5"] = by_player.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    return df


if __name__ == "__main__":
    conn = sqlite3.connect("data/euroleague.db")
    df = pd.read_sql_query(
        "SELECT * FROM boxscores WHERE Player_ID NOT IN ('Team', 'Total')",
        conn,
    )
    conn.close()

    df = add_rolling_form(df)

    sample = df[df["Player"] == "EDWARDS, CARSEN"]
    print(
        sample[
            ["Season", "Gamecode", "Player", "base_score", "rolling_3", "rolling_5"]
        ].to_string(index=False)
    )
