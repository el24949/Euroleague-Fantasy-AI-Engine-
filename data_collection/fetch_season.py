"""Fetch box scores for every regular-season game of a season and store them,
skipping games that are already in the database.

collect_season(season) is the reusable entry point -- pipeline/update_round.py
imports and calls it directly (rather than running this file as a script)
so a per-round refresh can target a live season without touching this
constant. This was a plain top-level script until that need came up: the
de-dup/fetch/store logic itself is unchanged, just moved into a function
parameterized on season instead of hardcoded to module-level SEASON. The
__main__ block below preserves the original standalone behavior (running
against SEASON, still 2023 by default) for anyone invoking this file
directly, e.g. `python -m data_collection.fetch_season`.
"""

import os
import sqlite3
import time

import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.schedule import Schedule

SEASON = 2023
DELAY_SECONDS = 1
DB_PATH = "data/euroleague.db"


def collect_season(season, delay_seconds=DELAY_SECONDS, db_path=DB_PATH):
    """Fetch and store every regular-season game's box score for `season`
    not already in the DB (de-dup guard: (Season, Gamecode) pairs already
    present are skipped, never re-fetched). Returns a summary dict --
    {"games_saved", "rows_saved", "rounds_scanned", "new_gamecodes"} --
    so a caller (pipeline/update_round.py) can report exactly what changed
    without scraping printed text."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path)

    existing_games = set()
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='boxscores'"
    ).fetchone()
    if table_exists:
        existing_df = pd.read_sql("SELECT DISTINCT Season, Gamecode FROM boxscores", conn)
        existing_games = set(zip(existing_df["Season"], existing_df["Gamecode"]))

    schedule = Schedule()
    schedule_df = schedule.get_schedule(season)
    regular_season = schedule_df[schedule_df["round"] == "RS"]
    rounds = sorted(regular_season["gameday"].unique())

    boxscore = BoxScoreData()
    total_games_saved = 0
    total_rows_saved = 0
    new_gamecodes = []

    for round_number in rounds:
        round_games = regular_season[regular_season["gameday"] == round_number]
        gamecodes = round_games["game"].astype(int).tolist()

        round_dfs = []
        for gamecode in gamecodes:
            if (season, gamecode) in existing_games:
                continue
            game_df = boxscore.get_players_boxscore_stats(season, gamecode)
            round_dfs.append(game_df)
            existing_games.add((season, gamecode))
            new_gamecodes.append(gamecode)
            time.sleep(delay_seconds)

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
    return {
        "games_saved": total_games_saved,
        "rows_saved": total_rows_saved,
        "rounds_scanned": len(rounds),
        "new_gamecodes": new_gamecodes,
    }


if __name__ == "__main__":
    collect_season(SEASON)
