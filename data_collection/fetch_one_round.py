"""Fetch box scores for every game in one round and append them to the database."""

import os
import sqlite3
import time

import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.schedule import Schedule

SEASON = 2023
ROUND = 1
DELAY_SECONDS = 1

schedule = Schedule()
schedule_df = schedule.get_schedule(SEASON)
round_games = schedule_df[schedule_df["gameday"] == ROUND]
gamecodes = round_games["game"].astype(int).tolist()

boxscore = BoxScoreData()
game_dfs = []
for gamecode in gamecodes:
    game_df = boxscore.get_players_boxscore_stats(SEASON, gamecode)
    game_dfs.append(game_df)
    time.sleep(DELAY_SECONDS)

combined_df = pd.concat(game_dfs, ignore_index=True)

os.makedirs("data", exist_ok=True)

conn = sqlite3.connect("data/euroleague.db")
combined_df.to_sql("boxscores", conn, if_exists="append", index=False)
conn.close()

print(
    f"Saved {len(gamecodes)} games / {len(combined_df)} rows "
    "to data/euroleague.db (table: boxscores)"
)
