"""Fetch player positions (Guards/Forwards/Centers) for every 2023 game and
store a Player_ID -> position lookup in data/euroleague.db.

Replaces the earlier leaderboard-based approach entirely. That endpoint
(PlayerStats.get_player_stats_leaders_single_season) capped at ~199 players
league-wide regardless of top_n and silently missed real rotation players
(Mirotic, Lucic, De Colo all confirmed missing, despite 300-550+ minutes
played). The real source is GameStats.get_game_stats(season, gamecode):
each team's `players` list embeds `position`/`positionName` directly on
every player who appeared in that game's box score -- the same per-game
granularity as our existing boxscores collector, so coverage should match
it instead of being capped.

Since every game must be scanned regardless (to detect players whose
position looks inconsistent across games), there's no fetch-cost reason to
skip already-fetched games the way fetch_season.py does for box scores.
So the de-dup guard here is a full replace of this season's rows on every
run (delete then insert) rather than skip-if-exists -- that also ensures
stale rows from the old leaderboard-based run never linger mixed in with
the new source.
"""

import os
import sqlite3
import time

import pandas as pd
from euroleague_api.game_stats import GameStats

SEASON = 2025
DB_PATH = "data/euroleague.db"
DELAY_SECONDS = 1
POSITION_NAME_TO_CODE = {"Guard": "G", "Forward": "F", "Center": "C"}

os.makedirs("data", exist_ok=True)

conn = sqlite3.connect(DB_PATH)

gamecodes = pd.read_sql_query(
    "SELECT DISTINCT Gamecode FROM boxscores WHERE Season = ?", conn, params=[SEASON]
)["Gamecode"].tolist()

game_stats = GameStats()

records = []
for gamecode in gamecodes:
    df = game_stats.get_game_stats(SEASON, gamecode)
    for side in ["local", "road"]:
        for entry in df.loc[0, f"{side}.players"]:
            person = entry["player"]["person"]
            position_name = entry["player"]["positionName"]
            if position_name not in POSITION_NAME_TO_CODE:
                raise ValueError(
                    f"Unknown positionName {position_name!r} for player "
                    f"{person['name']} (gamecode {gamecode}). "
                    f"Known values: {sorted(POSITION_NAME_TO_CODE)}"
                )
            records.append(
                {
                    "Player_ID": person["code"].strip(),
                    "Player": person["name"],
                    "Team": entry["player"]["club"]["code"],
                    "Position": POSITION_NAME_TO_CODE[position_name],
                    "Gamecode": gamecode,
                }
            )
    time.sleep(DELAY_SECONDS)

records_df = pd.DataFrame(records)

# --- conflict detection: don't silently overwrite disagreeing positions ---
positions_per_player = records_df.groupby("Player_ID")["Position"].unique()
conflicting_ids = set(positions_per_player[positions_per_player.apply(len) > 1].index)

conflicts_df = records_df[records_df["Player_ID"].isin(conflicting_ids)]
clean_df = records_df[~records_df["Player_ID"].isin(conflicting_ids)]

positions_df = clean_df.drop_duplicates(subset=["Player_ID"])[
    ["Player_ID", "Player", "Position"]
].copy()
positions_df["Season"] = SEASON
positions_df = positions_df[["Season", "Player_ID", "Player", "Position"]]

# --- store: delete this season's rows, then insert the fresh set ---
table_exists = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='positions'"
).fetchone()
if table_exists:
    conn.execute("DELETE FROM positions WHERE Season = ?", (SEASON,))
    conn.commit()

positions_df.to_sql("positions", conn, if_exists="append", index=False)
print(f"Saved {len(positions_df)} positions to data/euroleague.db (table: positions)")

# --- conflict report ---
if conflicting_ids:
    print(f"\n{len(conflicting_ids)} players had conflicting positions across games "
          "(excluded from the positions table):")
    for player_id in sorted(conflicting_ids):
        rows = conflicts_df[conflicts_df["Player_ID"] == player_id]
        name = rows["Player"].iloc[0]
        seen = rows[["Position", "Team", "Gamecode"]].drop_duplicates().to_dict("records")
        print(f"  {name} ({player_id}): {seen}")
else:
    print("\nNo conflicting positions found.")

# --- coverage report ---
boxscore_players = pd.read_sql_query(
    "SELECT DISTINCT Player_ID, Player FROM boxscores "
    "WHERE Season = ? AND Player_ID NOT IN ('Team', 'Total')",
    conn,
    params=[SEASON],
)
boxscore_players["Player_ID_normalized"] = (
    boxscore_players["Player_ID"].str.strip().str.lstrip("P")
)

matched_ids = set(positions_df["Player_ID"])
boxscore_players["has_position"] = boxscore_players["Player_ID_normalized"].isin(matched_ids)

total = len(boxscore_players)
matched = int(boxscore_players["has_position"].sum())
unmatched_players = boxscore_players[~boxscore_players["has_position"]]

print(f"\nSeason {SEASON} boxscores: {total} distinct players")
print(f"Matched to a position: {matched}")
print(f"NOT matched: {len(unmatched_players)}")
if len(unmatched_players):
    print(unmatched_players[["Player"]].to_string(index=False))

conn.close()
