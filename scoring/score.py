"""Season-aware fantasy score: base score plus the captain multiplier."""

import sqlite3

import pandas as pd

from scoring.base_score import base_fantasy_score
from scoring.season_rules import get_rules


def fantasy_score(row, season, is_captain=False):
    score = base_fantasy_score(row)
    if is_captain:
        score *= get_rules(season)["captain_multiplier"]
    return score


if __name__ == "__main__":
    conn = sqlite3.connect("data/euroleague.db")
    df = pd.read_sql_query("SELECT * FROM boxscores LIMIT 10", conn)
    conn.close()

    df["base_score"] = df.apply(base_fantasy_score, axis=1)
    df["captain_score"] = df.apply(
        lambda row: fantasy_score(row, row["Season"], is_captain=True), axis=1
    )

    print(df[["Player", "Season", "base_score", "captain_score"]].to_string(index=False))
