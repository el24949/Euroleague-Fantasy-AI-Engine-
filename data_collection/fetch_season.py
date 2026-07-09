"""Fetch box scores for every regular-season game of a season and store them,
skipping games that are already in the database."""

import os
import sqlite3
import time

import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.schedule import Schedule

SEASON = 2023
DELAY_SECONDS = 1
DB_PATH = "data/euroleague.db"

os.makedirs("data", exist_ok=True)

conn = sqlite3.connect(DB_PATH)

existing_games = set()
table_exists = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='boxscores'"
).fetchone()
if table_exists:
    existing_df = pd.read_sql("SELECT DISTINCT Season, Gamecode FROM boxscores", conn)
    existing_games = set(zip(existing_df["Season"], existing_df["Gamecode"]))

schedule = Schedule()
schedule_df = schedule.get_schedule(SEASON)
regular_season = schedule_df[schedule_df["round"] == "RS"]
rounds = sorted(regular_season["gameday"].unique())

boxscore = BoxScoreData()
total_games_saved = 0
total_rows_saved = 0

for round_number in rounds:
    round_games = regular_season[regular_season["gameday"] == round_number]
    gamecodes = round_games["game"].astype(int).tolist()

    round_dfs = []
    for gamecode in gamecodes:
        if (SEASON, gamecode) in existing_games:
            continue
        game_df = boxscore.get_players_boxscore_stats(SEASON, gamecode)
        round_dfs.append(game_df)
        existing_games.add((SEASON, gamecode))
        time.sleep(DELAY_SECONDS)

    if round_dfs:
        round_combined = pd.concat(round_dfs, ignore_index=True)
        round_combined.to_sql("boxscores", conn, if_exists="append", index=False)
        total_games_saved += len(round_dfs)
        total_rows_saved += len(round_combined)
        print(f"Round {round_number}: saved {len(round_dfs)} new games")
    else:
        print(f"Round {round_number}: no new games (already stored)")

conn.close()

print(
    f"Done: {total_games_saved} games / {total_rows_saved} rows stored "
    f"across {len(rounds)} regular-season rounds"
)
