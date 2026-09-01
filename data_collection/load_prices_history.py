"""Load fantasy prices PER ROUND into a history table, instead of the single
current-snapshot that load_prices.py reads.

Reuses load_prices.load_prices() (data_collection/load_prices.py) directly
for the actual sheet-parsing -- ID normalization ('P' + zfill for numeric
IDs, short codes used as-is), the Head-Coach/missing-Position/placeholder-ID
splits -- nothing about that parsing is reimplemented here. This file only
adds two things load_prices.py doesn't do: (1) a round dimension, so each
round's quotations are stored as history rather than overwriting one
snapshot, and (2) a small manual classification of placeholder-ID rows
that load_prices.py excludes wholesale -- some are confirmed duplicates of
an already-priced player (dropped) and some are genuinely unrecoverable
(reported, not silently dropped) -- see KNOWN_DUPLICATE_PLACEHOLDER_ROWS
below.

FOLDER LAYOUT: data/prices/round_NN/players_stats.xlsx, one subfolder per
round, each holding that round's price export in the same format
load_prices.py already parses. Discovered by globbing PRICES_ROOT rather
than a hardcoded round list, so dropping in a new round_NN folder is
enough -- no code change needed.

KNOWN_DUPLICATE_PLACEHOLDER_ROWS, WHY IT EXISTS: load_prices.py's own
"placeholder_id" exclusion (ID == "-") was built assuming that placeholder
is coach-only, but a handful of real (non-coach, real-Position) players
share it too -- confirmed by inspecting
data/prices/round_39/players_stats.xlsx directly. Of those, Ercan Osmani
(team code EFS) turned out NOT to be a missing player at all: he appears
TWICE in that file -- once correctly (ID 12710, Position Forward, real
season stats: Quotation 9.7, FPT 11.8) and once as a corrupted duplicate
(ID "-", name misspelled "Ercann", Position Center, Quotation 6.5, FPT
0.0 -- a stale/blank row, not a second real player). So this placeholder
row is DROPPED, not merged in as a new price -- the real one is already
present. Dalibor Ilic and Thirdy Ravena, by contrast, appear only ONCE
each, with FPT 0.0 (they never played a Season-2025 game -- verified
against boxscores, zero rows either), so there's no duplicate to point at
and no ground truth to assign them a real ID; they stay excluded and are
reported by name, not silently dropped, so a future round where they do
have real data can be revisited. This map is deliberately small and
manual: each entry records what a specific placeholder row actually is
(a confirmed duplicate of a specific real Player_ID), not a guess, and
load_prices_for_round asserts that real Player_ID is genuinely present
in the round's parsed prices before dropping the duplicate -- so a future
round where that assumption breaks fails loudly instead of silently
losing a player's price.

DE-DUP GUARD: existing (Season, Round, Player_ID) keys already in the
`prices` table are skipped on every run, the same skip-based pattern
fetch_season.py's collect_season() uses for boxscores -- re-running this
file (or reprocessing a round whose file hasn't changed) never creates
duplicate rows. Re-running after that round's file WAS edited/replaced
does not retroactively update already-stored rows (same limitation as
fetch_season.py's own de-dup guard); that's out of scope for this slice.
"""

import glob
import os
import re
import sqlite3

import pandas as pd

from data_collection.load_prices import load_prices

DB_PATH = "data/euroleague.db"
PRICES_ROOT = "data/prices"
SEASON = 2025

# (Surname, price-sheet Team code) -> the real Player_ID this placeholder
# row is a confirmed corrupted DUPLICATE of. See module docstring's
# KNOWN_DUPLICATE_PLACEHOLDER_ROWS section for how each entry was verified.
KNOWN_DUPLICATE_PLACEHOLDER_ROWS = {
    ("Osmani", "EFS"): "P012710",
}

PRICE_COLUMNS = ["Player_ID", "Name", "Surname", "Position", "Team", "Quotation", "Plus"]


def discover_round_files(prices_root=PRICES_ROOT):
    """Every round_NN/players_stats.xlsx found under prices_root, as a
    sorted list of round numbers. Fails loudly on a folder that doesn't
    match the expected naming, rather than silently skipping it."""
    pattern = os.path.join(prices_root, "round_*", "players_stats.xlsx")
    rounds = []
    for path in sorted(glob.glob(pattern)):
        folder = os.path.basename(os.path.dirname(path))
        match = re.fullmatch(r"round_(\d+)", folder)
        if not match:
            raise ValueError(
                f"Price folder {folder!r} under {prices_root} doesn't match "
                f"the expected 'round_NN' naming"
            )
        rounds.append(int(match.group(1)))
    return sorted(rounds)


def classify_placeholder_id_rows(excluded_placeholder_id, priced_ids):
    """Split load_prices()'s placeholder-ID ('-') exclusions into known
    duplicates (see KNOWN_DUPLICATE_PLACEHOLDER_ROWS -- dropped, since the
    real row is already in prices_df) and genuinely unrecoverable rows
    (reported by name, not silently dropped). Fails loudly if a row claims
    to duplicate a Player_ID that isn't actually in priced_ids -- that
    would mean the "real" row is missing too, which needs a person to look
    at, not a silent drop. Returns (known_duplicates_df, unrecoverable_df).
    """
    df = excluded_placeholder_id.copy()
    keys = list(zip(df["Surname"], df["Team"]))
    df["duplicate_of"] = [KNOWN_DUPLICATE_PLACEHOLDER_ROWS.get(key) for key in keys]

    known_duplicates = df[df["duplicate_of"].notna()]
    missing_target = known_duplicates[~known_duplicates["duplicate_of"].isin(priced_ids)]
    if len(missing_target):
        raise ValueError(
            "Placeholder-ID row(s) claim to duplicate a Player_ID that isn't "
            f"in this round's parsed prices -- the 'real' row may be missing "
            f"too: {missing_target[['Surname', 'Team', 'duplicate_of']].to_dict('records')}"
        )

    unrecoverable = df[df["duplicate_of"].isna()]
    return known_duplicates, unrecoverable


def load_prices_for_round(round_number, season=SEASON, prices_root=PRICES_ROOT):
    """Parse one round's price file (via load_prices.load_prices(), plus
    the placeholder-ID classification above), tag it with Season/Round, and
    return (prices_df, coverage) -- coverage is a plain dict, not a printed
    report, so refresh_price_history() can format it however it needs."""
    path = os.path.join(prices_root, f"round_{round_number:02d}", "players_stats.xlsx")
    prices_df, excluded = load_prices(path)

    known_duplicates, unrecoverable = classify_placeholder_id_rows(
        excluded["placeholder_id"], priced_ids=set(prices_df["Player_ID"])
    )

    prices_df = prices_df.copy()
    prices_df["Season"] = season
    prices_df["Round"] = round_number

    coverage = {
        "round": round_number,
        "path": path,
        "usable": len(prices_df),
        "head_coaches_excluded": len(excluded["head_coaches"]),
        "missing_position_excluded": len(excluded["missing_position"]),
        "placeholder_id_known_duplicates": len(known_duplicates),
        "placeholder_id_unrecovered_names": unrecoverable[["Name", "Surname", "Team"]].to_dict(
            "records"
        ),
    }
    return prices_df, coverage


def store_prices(prices_df, db_path=DB_PATH):
    """Append only rows whose (Season, Round, Player_ID) isn't already in
    the `prices` table. Returns the number of rows actually inserted."""
    conn = sqlite3.connect(db_path)
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prices'"
    ).fetchone()

    existing_keys = set()
    if table_exists:
        existing_df = pd.read_sql("SELECT Season, Round, Player_ID FROM prices", conn)
        existing_keys = set(zip(existing_df["Season"], existing_df["Round"], existing_df["Player_ID"]))

    is_new = [
        (season, round_number, player_id) not in existing_keys
        for season, round_number, player_id in zip(
            prices_df["Season"], prices_df["Round"], prices_df["Player_ID"]
        )
    ]
    new_rows = prices_df[is_new]

    if len(new_rows):
        columns = ["Season", "Round"] + PRICE_COLUMNS
        new_rows[columns].to_sql("prices", conn, if_exists="append", index=False)

    conn.close()
    return len(new_rows)


def refresh_price_history(season=SEASON, prices_root=PRICES_ROOT, db_path=DB_PATH):
    """Discover every round_NN price file under prices_root, load and store
    each into the `prices` table, and return a list of per-round coverage
    dicts (also printed, round by round, as this runs)."""
    rounds = discover_round_files(prices_root)
    if not rounds:
        print(f"No round price files found under {prices_root}/round_NN/players_stats.xlsx")
        return []

    summaries = []
    for round_number in rounds:
        prices_df, coverage = load_prices_for_round(round_number, season=season, prices_root=prices_root)
        newly_stored = store_prices(prices_df, db_path=db_path)
        coverage["rows_newly_stored"] = newly_stored
        coverage["rows_already_stored"] = coverage["usable"] - newly_stored
        summaries.append(coverage)

        print(f"\nRound {round_number} ({coverage['path']}):")
        print(f"  Usable prices parsed: {coverage['usable']}")
        print(
            f"  Excluded -- head coaches: {coverage['head_coaches_excluded']}, "
            f"missing position: {coverage['missing_position_excluded']}"
        )
        print(
            f"  Placeholder-ID rows: {coverage['placeholder_id_known_duplicates']} dropped "
            f"as known duplicates, {len(coverage['placeholder_id_unrecovered_names'])} still unrecovered"
        )
        for row in coverage["placeholder_id_unrecovered_names"]:
            print(f"    unrecovered: {row['Name']} {row['Surname']} ({row['Team']})")
        print(
            f"  Stored: {coverage['rows_newly_stored']} new / "
            f"{coverage['rows_already_stored']} already in DB"
        )

    return summaries


if __name__ == "__main__":
    summaries = refresh_price_history()
    total_new = sum(s["rows_newly_stored"] for s in summaries)
    total_rows = sum(s["usable"] for s in summaries)
    print(
        f"\nDone: {total_new} new row(s) stored across {len(summaries)} round file(s) "
        f"({total_rows} total usable prices parsed)"
    )
