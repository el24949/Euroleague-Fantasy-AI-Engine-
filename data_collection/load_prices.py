"""Load fantasy prices from data/prices/players_stats.xlsx.

The workbook's ID column doesn't match boxscores.Player_ID directly: it's a
bare integer (e.g. 5985) where boxscores uses a zero-padded 'P' code (e.g.
P005985). load_prices() rebuilds that code so the result can be joined
straight onto boxscores.Player_ID (after the caller strips it -- see
build_player_id's docstring below) elsewhere in the project, e.g.
optimizer/pick_lineup_ilp.py.

Run as `python -m data_collection.load_prices` for a standalone coverage
report against Season=2025 boxscores; this module does not write to the
database.
"""

import sqlite3

import pandas as pd

DB_PATH = "data/euroleague.db"
PRICES_PATH = "data/prices/players_stats.xlsx"
SEASON = 2025


def build_player_id(id_value):
    # boxscores.Player_ID (sourced via euroleague-api) is usually 'P' + a
    # 6-digit zero-padded numeric code, but a handful of players only ever
    # got a short non-numeric code from EuroLeague (e.g. 'PKLT' for Thomas
    # Heurtel) -- confirmed by looking those players up directly in
    # boxscores. The price sheet's ID column carries the same two shapes, so
    # only the numeric case gets zero-padded; the short code is used as-is.
    numeric = pd.to_numeric(id_value, errors="coerce")
    if pd.notna(numeric):
        return "P" + str(int(numeric)).zfill(6)
    return "P" + str(id_value)


def load_prices(prices_path=PRICES_PATH):
    raw = pd.read_excel(prices_path)

    # The sheet ends with a fully-blank trailer row and a footnote row
    # ("Reference matchdays: ...") that both carry no Name -- drop them
    # before anything else so they can't pollute the ID/Position checks
    # below.
    raw = raw[raw["Name"].notna()].copy()

    # --- flag: Head Coaches ---
    # Coaches share the literal placeholder ID "-" (which is why ID looked
    # like it had 22 duplicates before this split) and aren't selectable in
    # Classic mode, so they're excluded from the price table.
    is_coach = raw["Position"] == "Head Coach"
    coach_rows = raw[is_coach]
    players = raw[~is_coach].copy()

    # --- flag: missing Position ---
    no_position = players[players["Position"].isna()]
    players = players[players["Position"].notna()].copy()

    # --- flag: placeholder ID ---
    # The literal "-" (Head Coaches' shared placeholder) is the one ID value
    # that can't be turned into anything; drop and report any that slipped
    # through the Head Coach filter above.
    bad_id = players[players["ID"] == "-"]
    players = players[players["ID"] != "-"].copy()

    players["Player_ID"] = players["ID"].apply(build_player_id)

    dup_ids = players[players["Player_ID"].duplicated(keep=False)]
    if len(dup_ids):
        raise ValueError(
            f"{dup_ids['Player_ID'].nunique()} Player_ID collisions after "
            f"transform -- refusing to build an ambiguous price table:\n"
            f"{dup_ids[['Player_ID', 'Name', 'Surname', 'Team']].to_string(index=False)}"
        )

    prices_df = players[
        ["Player_ID", "Name", "Surname", "Position", "Team", "Quotation", "Plus"]
    ]
    excluded = {
        "head_coaches": coach_rows,
        "missing_position": no_position,
        "placeholder_id": bad_id,
    }
    return prices_df, excluded


if __name__ == "__main__":
    prices_df, excluded = load_prices()

    print(f"Loaded rows from {PRICES_PATH}")
    print(f"  Head Coaches (excluded): {len(excluded['head_coaches'])}")
    print(f"  Missing Position (excluded): {len(excluded['missing_position'])}")
    print(f"  Placeholder ID (excluded): {len(excluded['placeholder_id'])}")
    if len(excluded["placeholder_id"]):
        print(
            excluded["placeholder_id"][["ID", "Name", "Surname", "Position", "Team"]]
            .to_string(index=False)
        )
    print(f"  -> {len(prices_df)} usable player prices")

    # --- coverage check against Season=2025 boxscores ---
    conn = sqlite3.connect(DB_PATH)

    boxscores = pd.read_sql_query(
        "SELECT Player_ID, Player, Minutes FROM boxscores "
        "WHERE Season = ? AND Player_ID NOT IN ('Team', 'Total')",
        conn,
        params=[SEASON],
    )
    conn.close()

    # boxscores.Player_ID is a fixed-width source field padded with trailing
    # spaces (e.g. 'P005985   ') -- strip it or every join below silently
    # misses.
    boxscores["Player_ID"] = boxscores["Player_ID"].str.strip()

    # Minutes is 'MM:SS' for players who played and the literal 'DNP' for
    # those who didn't -- DNP contributes 0 to the per-player minutes total
    # below.
    def parse_minutes(value):
        if value == "DNP":
            return 0.0
        mins, secs = value.split(":")
        return int(mins) + int(secs) / 60

    boxscores["minutes_played"] = boxscores["Minutes"].apply(parse_minutes)

    per_player = boxscores.groupby(["Player_ID", "Player"]).agg(
        games=("Minutes", "size"),
        total_minutes=("minutes_played", "sum"),
    ).reset_index()
    per_player["avg_minutes"] = per_player["total_minutes"] / per_player["games"]

    priced_ids = set(prices_df["Player_ID"])
    per_player["has_price"] = per_player["Player_ID"].isin(priced_ids)

    total = len(per_player)
    matched = int(per_player["has_price"].sum())
    unmatched = per_player[~per_player["has_price"]].sort_values(
        "total_minutes", ascending=False
    )

    print(f"\nSeason {SEASON} boxscores: {total} distinct players")
    print(f"Matched to a price: {matched}")
    print(f"NOT matched: {len(unmatched)}")

    if len(unmatched):
        print("\nUnmatched players, ranked by total minutes played (top 20):")
        print(
            unmatched[["Player", "games", "total_minutes", "avg_minutes"]]
            .head(20)
            .to_string(index=False)
        )
