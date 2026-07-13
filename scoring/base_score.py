"""Compute a player's base EuroLeague Fantasy score from one box-score row.

This is season-independent: it's just the +1/-1 counting stats from the
official rules. Season-specific multipliers (e.g. captain x1.5 vs x2) are
applied on top of this later, elsewhere.
"""

import sqlite3

import pandas as pd


def base_fantasy_score(row):
    missed_fg2 = row["FieldGoalsAttempted2"] - row["FieldGoalsMade2"]
    missed_fg3 = row["FieldGoalsAttempted3"] - row["FieldGoalsMade3"]
    missed_ft = row["FreeThrowsAttempted"] - row["FreeThrowsMade"]

    positives = (
        row["Points"]
        + row["OffensiveRebounds"]
        + row["DefensiveRebounds"]
        + row["Assistances"]
        + row["Steals"]
        + row["BlocksFavour"]
        + row["FoulsReceived"]
    )

    negatives = (
        row["Turnovers"]
        + row["BlocksAgainst"]
        + row["FoulsCommited"]
        + missed_fg2
        + missed_fg3
        + missed_ft
    )

    return positives - negatives


if __name__ == "__main__":
    conn = sqlite3.connect("data/euroleague.db")
    df = pd.read_sql_query("SELECT * FROM boxscores LIMIT 10", conn)
    conn.close()

    df["base_score"] = df.apply(base_fantasy_score, axis=1)

    print(df[["Player", "Points", "base_score"]].to_string(index=False))
