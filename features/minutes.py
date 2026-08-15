"""Per-player rolling minutes features: recent playing-time trend.

Rolling averages are shifted so a game's own minutes are never included
in its own rolling window (no data leakage) -- only strictly earlier
games in that player's chronological history are used.
"""

import sqlite3

import pandas as pd


def minutes_to_float(minutes_str):
    if minutes_str is None or minutes_str == "DNP":
        return 0.0
    mm, ss = minutes_str.split(":")
    return int(mm) + int(ss) / 60


def add_rolling_minutes(df):
    df["minutes_numeric"] = df["Minutes"].apply(minutes_to_float)

    df = df.sort_values(["Player_ID", "Season", "Gamecode"]).reset_index(drop=True)

    # Gamecode resets each season, so grouping on Player_ID alone would let
    # a rolling window reach across the season boundary (see
    # features/rolling_form.py for the same fix). [Season, Player_ID] gives
    # each player a fresh group per season.
    by_player = df.groupby(["Season", "Player_ID"])["minutes_numeric"]
    df["rolling_3_minutes"] = by_player.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling_5_minutes"] = by_player.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    return df


if __name__ == "__main__":
    conn = sqlite3.connect("data/euroleague.db")
    df = pd.read_sql_query(
        "SELECT Season, Gamecode, Player_ID, Player, Minutes FROM boxscores "
        "WHERE Player_ID NOT IN ('Team', 'Total')",
        conn,
    )
    conn.close()

    df = add_rolling_minutes(df)

    sample = df[df["Player"] == "EDWARDS, CARSEN"]
    print(
        sample[
            [
                "Season",
                "Gamecode",
                "Player",
                "minutes_numeric",
                "rolling_3_minutes",
                "rolling_5_minutes",
            ]
        ].to_string(index=False)
    )
