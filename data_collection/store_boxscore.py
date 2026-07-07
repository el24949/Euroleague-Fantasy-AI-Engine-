"""Fetch one game's player box score and save it into a local SQLite database."""

import os
import sqlite3

from euroleague_api.boxscore_data import BoxScoreData

SEASON = 2023
GAMECODE = 1

boxscore = BoxScoreData()
df = boxscore.get_players_boxscore_stats(SEASON, GAMECODE)

os.makedirs("data", exist_ok=True)

conn = sqlite3.connect("data/euroleague.db")
df.to_sql("boxscores", conn, if_exists="replace", index=False)
conn.close()

print(f"Saved {len(df)} rows to data/euroleague.db (table: boxscores)")
