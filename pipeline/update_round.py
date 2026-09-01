"""Per-round data-refresh orchestrator -- the go-live update routine.

Run this before picking a team each round to bring data/euroleague.db up to
date for SEASON (2025, the current working season). In order:
  1. box-score collector (data_collection.fetch_season.collect_season)
  2. positions collector (data_collection.fetch_positions, run via import)
  3. turn-tag refresh (features.turn_tags.build_turn_lookup)
then a status summary: latest round/gamecode present in the DB, total
games, and which round is 'next/upcoming' -- the one still needing a team.

DELIBERATELY NOT IN THIS SLICE: price refresh, running the optimizer. This
is the data-refresh skeleton only -- optimizer/pick_lineup_ilp.py still
needs to be run separately once prices and this fresh data are in place.

WHERE THE MANUAL PRICE STEP BELONGS: unlike box scores/positions/turns,
there's no API for fantasy prices (login-gated) -- a person has to
download that round's price Excel from the EuroLeague Fantasy site by
hand and drop it at data/prices/round_NN/players_stats.xlsx before
data_collection/load_prices_history.py can load it (see that file's own
docstring). That has to happen BETWEEN this pipeline finishing and the
optimizer running -- this file can compute which round number NN is (the
next/upcoming round the status summary below already identifies) and say
so, but can't do the download itself. status_summary() prints that
reminder using the next_round it just computed, rather than leaving it as
a comment only visible in this docstring.

Reuses collect_season (data_collection/fetch_season.py), fetch_positions.py
(data_collection/fetch_positions.py, imported rather than called -- see
STEP 2 below), and build_turn_lookup/load_regular_season_schedule
(features/turn_tags.py) -- nothing here is recomputed.

STEP 1: collect_season(season) used to be fetch_season.py's entire
module-level script body, hardcoded to SEASON=2023 and executed on import.
It's now a season-parameterized function (see that file's own docstring for
why) specifically so this pipeline could import and call it against 2025
without touching the de-dup/fetch/store logic itself, which is unchanged.

STEP 2, WHY A PLAIN IMPORT: fetch_positions.py already targets this
pipeline's working season (its own module-level SEASON=2025 already
matches -- unlike fetch_season.py, it needed no season-parameter fix) but
isn't wrapped in a callable function, so importing IT IS running it. It's
also not skip-based like the box-score collector -- by design (see that
file's own module docstring), it deletes and re-inserts the season's whole
positions table every run, since every game must be re-scanned regardless
to catch inconsistent positions across games. So "newly added vs skipped"
doesn't map cleanly onto this step; it reports a before/after row-count
diff instead, to show whether the refresh actually changed anything. Uses
importlib to (re)run the module body on every call to run_positions_step,
not just the first import in the process -- a plain `import
data_collection.fetch_positions` would silently no-op on a second call in
the same process (Python caches modules), which would break the "repeatable"
part of a per-round refresh routine.

STEP 3, WHY THERE'S NO DB WRITE TO REPORT: build_turn_lookup returns an
in-memory dict only -- turn_tags.py never had a persistence layer to begin
with (see that file's own module docstring). "Refreshing" here means
re-deriving the lookup from the live schedule (so a reschedule that
happened since the last run is picked up), and reporting its current
shape -- not diffing against stored state, since none exists.
"""

import importlib
import sqlite3
import sys

import pandas as pd

from data_collection.fetch_season import collect_season
from features.turn_tags import build_turn_lookup, load_regular_season_schedule

SEASON = 2025
DB_PATH = "data/euroleague.db"


def get_positions_row_count(season, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM positions WHERE Season = ?", (season,)).fetchone()[0]
    conn.close()
    return count


def run_box_score_step(season):
    """Step 1: incremental box-score collection -- reports the actual
    new Gamecodes saved, from collect_season's own return value."""
    print(f"\n=== Step 1/3: box-score collector (Season {season}) ===")
    summary = collect_season(season)
    if summary["new_gamecodes"]:
        print(
            f"Newly added: {summary['games_saved']} game(s), Gamecodes "
            f"{sorted(summary['new_gamecodes'])}"
        )
    else:
        print(f"Newly added: 0 games -- all {summary['rounds_scanned']} rounds already stored")
    return summary


def run_positions_step(season):
    """Step 2: positions refresh -- full delete+reinsert every call (see
    module docstring), reported as a before/after row-count diff."""
    print(f"\n=== Step 2/3: positions collector (Season {season}) ===")
    before = get_positions_row_count(season)

    module_name = "data_collection.fetch_positions"
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)

    after = get_positions_row_count(season)
    print(
        f"Positions refresh is a full delete+reinsert every run, not "
        f"skip-based -- before: {before} rows, after: {after} rows "
        f"({'changed' if before != after else 'unchanged'})"
    )
    return {"before": before, "after": after}


def run_turn_tag_step(season):
    """Step 3: rebuild the turn lookup from the live schedule (no DB write --
    see module docstring). Reports the T1/T2/RESCHEDULED shape."""
    print(f"\n=== Step 3/3: turn-tag refresh (Season {season}) ===")
    turn_lookup = build_turn_lookup(season)
    turns = list(turn_lookup.values())
    t1, t2, rescheduled = turns.count("T1"), turns.count("T2"), turns.count("RESCHEDULED")
    print(
        f"Rebuilt from the live schedule (in-memory only, no DB write): "
        f"{len(turn_lookup)} games tagged (T1={t1}, T2={t2}, RESCHEDULED={rescheduled})"
    )
    return turn_lookup


def status_summary(season, db_path=DB_PATH):
    """Latest round/gamecode present, total games, and the next/upcoming
    round -- the first round with any game missing from the DB, i.e. the
    one a manager still needs to pick a team for."""
    print(f"\n=== Status summary (Season {season}) ===")
    schedule_df = load_regular_season_schedule(season)
    round_to_gamecodes = schedule_df.groupby("gameday")["game"].apply(set).to_dict()

    conn = sqlite3.connect(db_path)
    present_gamecodes = set(
        pd.read_sql_query(
            "SELECT DISTINCT Gamecode FROM boxscores WHERE Season = ?", conn, params=[season]
        )["Gamecode"]
    )
    conn.close()

    total_games = len(present_gamecodes)
    latest_gamecode = max(present_gamecodes) if present_gamecodes else None
    latest_round_with_any_data = None
    if latest_gamecode is not None:
        match = schedule_df.loc[schedule_df["game"] == latest_gamecode, "gameday"]
        latest_round_with_any_data = int(match.iloc[0]) if len(match) else None

    all_rounds = sorted(round_to_gamecodes)
    complete_rounds = [r for r in all_rounds if round_to_gamecodes[r] <= present_gamecodes]
    incomplete_rounds = [r for r in all_rounds if r not in complete_rounds]

    latest_complete_round = max(complete_rounds) if complete_rounds else None
    next_round = min(incomplete_rounds) if incomplete_rounds else None

    print(f"Total games in DB: {total_games} / {len(schedule_df)} scheduled regular-season games")
    print(f"Latest Gamecode present: {latest_gamecode} (round {latest_round_with_any_data})")
    print(f"Latest COMPLETE round in DB: {latest_complete_round}")
    if next_round is not None:
        missing = sorted(round_to_gamecodes[next_round] - present_gamecodes)
        print(
            f"Next/upcoming round (to pick a team for): {next_round} -- "
            f"{len(missing)} game(s) not yet in DB: {missing}"
        )
    else:
        print("Next/upcoming round: none -- every scheduled regular-season round is fully collected")

    # --- MANUAL STEP: prices have no API (login-gated), so this can only
    # --- name where the file belongs, not fetch it -- see module docstring's
    # --- "WHERE THE MANUAL PRICE STEP BELONGS" section ---
    print("\nManual step before running the optimizer:")
    if next_round is not None:
        print(
            f"  Download round {next_round}'s price Excel from the EuroLeague Fantasy "
            f"site and place it at data/prices/round_{next_round:02d}/players_stats.xlsx, "
            f"then run `python -m data_collection.load_prices_history` to load it into "
            f"the prices table (data_collection/load_prices_history.py)."
        )
    else:
        print("  N/A -- no upcoming round to price (every scheduled round is fully collected).")

    return {
        "total_games": total_games,
        "latest_gamecode": latest_gamecode,
        "latest_complete_round": latest_complete_round,
        "next_round": next_round,
    }


if __name__ == "__main__":
    print(f"Data refresh: Season {SEASON}")
    run_box_score_step(SEASON)
    run_positions_step(SEASON)
    run_turn_tag_step(SEASON)
    status_summary(SEASON)
