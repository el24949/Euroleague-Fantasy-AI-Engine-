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
from datetime import datetime

import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.schedule import Schedule

SEASON = 2023
DELAY_SECONDS = 1
DB_PATH = "data/euroleague.db"


def collect_season(season, delay_seconds=DELAY_SECONDS, db_path=DB_PATH):
    """Fetch and store every regular-season game's box score for `season`
    not already in the DB (de-dup guard: (Season, Gamecode) pairs already
    present are skipped, never re-fetched) AND already played (its
    scheduled kickoff is in the past -- see below). Returns a summary dict
    -- {"games_saved", "rows_saved", "rounds_scanned", "new_gamecodes",
    "skipped_gamecodes", "not_yet_scheduled_count"} -- so a caller
    (pipeline/update_round.py, app.py) can report exactly what changed
    without scraping printed text.

    FUTURE GAMES ARE NEVER REQUESTED: the Schedule data this function
    already loads has 'date' (e.g. 'Sep 24, 2026') + 'startime' (e.g.
    '20:15') columns giving each game's scheduled kickoff -- confirmed by
    inspection (euroleague_api.schedule.Schedule().get_schedule(season)):
    parses cleanly for all 380 Season-2026 rows with zero failures, and
    exactly matches that schedule's own boolean 'played' column (30
    past / 350 future, as of 2026-10-05) -- so it's a reliable "has this
    game actually happened" signal. A game whose kickoff hasn't happened
    yet is skipped WITHOUT ever calling the API for it (that's the ~6
    minute slowdown this fixes: hitting the API for 350 games that can't
    possibly have data) and is counted in not_yet_scheduled_count, NOT
    skipped_gamecodes -- "not played yet" isn't an error. A row whose
    date/time fails to parse (never seen in practice) is NOT skipped on
    that basis -- it's attempted anyway, so a schedule-parsing hiccup can
    never cause a game to go silently unrequested forever.

    A PAST game the API still has no/invalid data for (e.g. just finished,
    not published yet -- raises JSONDecodeError rather than returning an
    empty result) IS attempted, and is SKIPPED, not fatal, if it fails:
    nothing is written for it, and it's NOT added to existing_games, so
    the very next collect_season() call retries it instead of silently
    losing it forever. Its gamecode goes in skipped_gamecodes so the
    caller can say so instead of just going quiet about it."""
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
    regular_season = schedule_df[schedule_df["round"] == "RS"].copy()
    regular_season["scheduled_at"] = pd.to_datetime(
        regular_season["date"] + " " + regular_season["startime"],
        format="%b %d, %Y %H:%M",
        errors="coerce",
    )
    rounds = sorted(regular_season["gameday"].unique())

    boxscore = BoxScoreData()
    total_games_saved = 0
    total_rows_saved = 0
    new_gamecodes = []
    skipped_gamecodes = []
    not_yet_scheduled_count = 0
    now = datetime.now()

    for round_number in rounds:
        round_games = regular_season[regular_season["gameday"] == round_number]

        round_dfs = []
        for row in round_games.itertuples():
            gamecode = int(row.game)
            if (season, gamecode) in existing_games:
                continue
            if pd.notna(row.scheduled_at) and row.scheduled_at > now:
                not_yet_scheduled_count += 1
                continue
            try:
                game_df = boxscore.get_players_boxscore_stats(season, gamecode)
            except Exception as exc:
                # Not published yet (or otherwise unavailable) -- do NOT mark it in
                # existing_games, so it's retried on the next call instead of lost.
                print(f"Season {season} Gamecode {gamecode}: skipped ({exc})")
                skipped_gamecodes.append(gamecode)
                time.sleep(delay_seconds)
                continue
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
            print(f"Round {round_number}: no new games (already stored, not yet played, or skipped)")

    conn.close()

    print(
        f"Done: {total_games_saved} games / {total_rows_saved} rows stored across "
        f"{len(rounds)} regular-season rounds ({not_yet_scheduled_count} future game(s) "
        f"not attempted, {len(skipped_gamecodes)} attempted-but-skipped)"
    )
    return {
        "games_saved": total_games_saved,
        "rows_saved": total_rows_saved,
        "rounds_scanned": len(rounds),
        "new_gamecodes": new_gamecodes,
        "skipped_gamecodes": skipped_gamecodes,
        "not_yet_scheduled_count": not_yet_scheduled_count,
    }


if __name__ == "__main__":
    collect_season(SEASON)
