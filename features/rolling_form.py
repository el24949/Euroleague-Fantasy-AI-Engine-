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

    # Gamecode resets to 1 each season, so a player active in multiple
    # seasons has two chronologically-unrelated game logs sharing the same
    # numbering. Grouping on Player_ID alone would let a rolling window (or
    # the expanding season-to-date average below) reach across the season
    # boundary -- e.g. a player's first 2025 game inheriting form from their
    # last 2023 game. Keying on [Season, Player_ID] gives each player a
    # fresh group per season, so every window and the expanding mean reset
    # at the season boundary.
    by_player = df.groupby(["Season", "Player_ID"])["base_score"]
    df["rolling_3"] = by_player.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling_5"] = by_player.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["consistency_5"] = by_player.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).std()
    )

    season_to_date_avg = by_player.transform(lambda s: s.shift(1).expanding().mean())
    deviation_from_norm = df["base_score"] - season_to_date_avg
    df["distance_from_norm"] = deviation_from_norm.groupby(
        [df["Season"], df["Player_ID"]]
    ).shift(1)

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
            [
                "Season",
                "Gamecode",
                "Player",
                "base_score",
                "rolling_3",
                "rolling_5",
                "consistency_5",
                "distance_from_norm",
            ]
        ].to_string(index=False)
    )
