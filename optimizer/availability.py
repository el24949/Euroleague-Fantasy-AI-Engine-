"""Loads data/unavailable.csv (hand-edited: one row per player who is out or
doubtful for the upcoming round) and resolves each player_name to the
Player_ID used in the boxscores table / df_round, so pick_lineup_ilp.py can
drop those players from the candidate pool entirely.

Matching is case-insensitive against the boxscores.Player column, which is
formatted 'LASTNAME, FIRSTNAME' (see fetch_positions.py / pick_team.py for
the same convention). Fails loudly -- raises ValueError -- if a name matches
zero players (likely a typo or wrong format) or more than one distinct
Player_ID (an ambiguous name); this is a hand-edited override list, so a
silent skip would mean a player who should be excluded quietly stays in the
pool.
"""

import sqlite3

import pandas as pd

DB_PATH = "data/euroleague.db"
UNAVAILABLE_CSV = "data/unavailable.csv"


def load_unavailable(csv_path=UNAVAILABLE_CSV, db_path=DB_PATH):
    unavailable_df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)
    boxscore_players = pd.read_sql_query(
        "SELECT DISTINCT Player_ID, Player FROM boxscores", conn
    )
    conn.close()

    boxscore_players["Player_ID"] = boxscore_players["Player_ID"].str.strip()
    boxscore_players["Player_norm"] = boxscore_players["Player"].str.strip().str.upper()

    resolved = []
    for _, row in unavailable_df.iterrows():
        name = row["player_name"]
        name_norm = str(name).strip().upper()
        matches = boxscore_players[boxscore_players["Player_norm"] == name_norm]

        if len(matches) == 0:
            raise ValueError(
                f"data/unavailable.csv: {name!r} matched zero players in boxscores. "
                f"Check spelling and format -- Player is stored as 'LASTNAME, FIRSTNAME' "
                f"(e.g. 'LARKIN, SHANE')."
            )

        distinct_ids = matches["Player_ID"].unique()
        if len(distinct_ids) > 1:
            raise ValueError(
                f"data/unavailable.csv: {name!r} matched {len(distinct_ids)} different "
                f"Player_IDs in boxscores: {list(distinct_ids)}. Name is ambiguous -- "
                f"this loader has no way to disambiguate two players with the same "
                f"stored name."
            )

        resolved.append(
            {
                "player_name": name,
                "player_id": distinct_ids[0],
                "note": row["note"],
            }
        )

    return resolved
