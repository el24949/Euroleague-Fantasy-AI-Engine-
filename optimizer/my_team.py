"""Load, resolve, and validate the team-state input for the engine (row 7
of the data requirements doc): "what's currently on my roster", as opposed
to "what should the engine recommend". Two hand-edited CSVs:
  - data/my_team.csv -- 10 players (role: starter/6th_man/bench) + 1 coach,
    with a captain marker. Player names are typed by hand.
  - data/my_team_state.csv -- remaining_budget and trades_used, simple
    editable fields tracked separately from the roster.

load_my_team() is the reusable entry point other engine parts should
import to ask "what's my current team" -- returns a dict with a resolved
players DataFrame (Player_ID, position, price attached), a coach dict, and
the state dict, WITHOUT running the optimizer or changing anything about
it (that's explicitly out of scope for this slice).

REPOINTED TO THE 2026 PRICE FILE (this is the second version of this
loader): the first version resolved player names against Season 2026
boxscores, which was correct in principle but useless in practice --
Season 2026 has no box-score or position data at all yet (verified
directly against euroleague-api; no games have been played). PRICE_FILE
(data/prices/round_01/players_stats.xlsx -- a real preseason snapshot: 348
players, all 20 clubs, Quotation populated, FPT all 0.0 since no games
have been played, which is the CORRECT signature for a genuine pre-season
round-1 file) has real names/positions/prices/IDs right now, so player
NAME, POSITION, PRICE, and PLAYER_ID all come from this ONE file instead --
reused via data_collection.load_prices.load_prices() (its 'P'+zfill(6)
Player_ID construction and its head-coach/missing-position/placeholder-ID
splitting are used as-is, not reimplemented). COACH data comes from the
SAME price_path file as the players, via load_coach_pool() below
(load_prices(prices_path=price_path)'s excluded['head_coaches']), so a
coach's price can never come from a different snapshot than the players'.
It deliberately does NOT use optimizer/coach_ranking.load_coach_info(),
which reads the older generic data/prices/players_stats.xlsx and remaps
team codes through PRICE_SHEET_TEAM_CODE_MAP (2025-derived, missing the
2026 clubs BJK/FBT/MIL, so it would raise on this file). load_coach_pool's
team_code is therefore the price sheet's own club code; it is only
displayed / used to disambiguate a coach name, never for coach scoring.
(load_coach_info is still what coach_ranking.rank_coaches and
optimizer/pick_lineup_ilp.py use -- untouched by this loader.)

PLACEHOLDER-ID RECOVERY: load_prices() already splits out rows whose ID is
the literal placeholder '-' (excluded['placeholder_id']) -- confirmed
non-coach real players (e.g. Valanciunas in this file), just missing a
real ID in the price sheet. recover_placeholder_id() gets them a real
Player_ID by matching name + team against Season 2025 boxscores (the most
recent season with real rosters -- RECOVERY_SEASON below), translating the
price sheet's team code via PRICE_SHEET_TEAM_CODE_MAP (optimizer/
coach_ranking.py, reused as-is -- not re-derived here). This only runs for
a placeholder player the user actually typed into my_team.csv (not eagerly
for the whole price file), and fails loudly -- naming the player -- if the
team code isn't mapped, or if name+team matches zero or more than one
Season-2025 Player_ID.

VALIDATION (validate_roster): fails loudly, matching this project's
fail-loud style throughout, on: wrong row count, wrong role counts (must
be exactly 5 starter/1 6th_man/4 bench/1 coach), wrong position counts
(must match POSITION_REQUIREMENTS, i.e. 4G/4F/2C), not exactly one
captain, or a captain who isn't a starter (the same ASSUMPTION
optimizer/pick_lineup_ilp.py's own docstring flags as unverified against
the live app -- reused here for consistency, not re-derived). Position
counts are still checked against POSITION_REQUIREMENTS (optimizer/
pick_team.py, reused), even though positions no longer come from that
file's attach_positions()/the `positions` DB table.

NO OPTIMIZER CHANGES: this file only loads/validates/reports the CURRENT
team. Nothing here calls solve_lineup or feeds this roster into the
optimizer -- that's future work.
"""

import sqlite3

import pandas as pd

from data_collection.load_prices import load_prices
from optimizer.coach_ranking import PRICE_SHEET_TEAM_CODE_MAP
from optimizer.pick_team import POSITION_REQUIREMENTS

DB_PATH = "data/euroleague.db"
MY_TEAM_CSV = "data/my_team.csv"
MY_TEAM_STATE_CSV = "data/my_team_state.csv"
SEASON = 2026
PRICE_ROUND = 1
PRICE_FILE = f"data/prices/round_{PRICE_ROUND:02d}/players_stats.xlsx"

# Season 2026 boxscores are empty (no games played), so a placeholder-ID
# player can't be recovered from SEASON itself -- 2025 is the most recent
# season with real rosters to match name+team against.
RECOVERY_SEASON = 2025

ROLE_REQUIREMENTS = {"starter": 5, "6th_man": 1, "bench": 4}
VALID_ROLES = set(ROLE_REQUIREMENTS) | {"coach"}
CAPTAIN_TRUE_VALUES = {"Y", "YES", "TRUE", "1"}

# The price sheet's Position column spells these out; boxscores/
# POSITION_REQUIREMENTS use the short G/F/C codes. Not imported from
# data_collection/fetch_positions.py -- that module runs a full
# fetch-and-DB-write as a side effect of being imported (see
# pipeline/update_round.py's own docstring on why it's run via importlib,
# never `import`ed directly), so duplicating this fixed 3-entry mapping
# here is deliberate, not an oversight.
POSITION_NAME_TO_CODE = {"Guard": "G", "Forward": "F", "Center": "C"}


def _is_captain(value):
    return str(value).strip().upper() in CAPTAIN_TRUE_VALUES


def load_roster_csv(csv_path=MY_TEAM_CSV):
    """Read data/my_team.csv (comment lines starting with '#' skipped),
    and fail loudly on the two mistakes a hand-edited file can't recover
    from silently: a role value outside the known set, or a row the
    template shipped with that's still blank (never filled in)."""
    df = pd.read_csv(csv_path, comment="#", dtype=str)
    df["player_name"] = df["player_name"].fillna("").str.strip()
    df["role"] = df["role"].fillna("").str.strip()
    df["captain"] = df["captain"].fillna("")

    bad_roles = set(df["role"]) - VALID_ROLES
    if bad_roles:
        raise ValueError(
            f"{csv_path}: unknown role value(s) {sorted(bad_roles)} -- "
            f"must be one of {sorted(VALID_ROLES)}"
        )

    blank = df["player_name"] == ""
    if blank.any():
        raise ValueError(
            f"{csv_path} has {blank.sum()} row(s) with no player_name filled "
            f"in yet (role(s): {df.loc[blank, 'role'].tolist()}) -- fill in "
            f"the template before loading."
        )
    return df


def load_boxscore_players(season, db_path=DB_PATH):
    """Distinct (Player_ID, Player, Team) for SEASON, normalized exactly
    like optimizer/availability.py's load_unavailable() does. Used only by
    recover_placeholder_id() below, against RECOVERY_SEASON."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT DISTINCT Player_ID, Player, Team FROM boxscores "
        "WHERE Season = ? AND Player_ID NOT IN ('Team', 'Total')",
        conn,
        params=[season],
    )
    conn.close()
    df["Player_ID"] = df["Player_ID"].str.strip()
    df["Player_norm"] = df["Player"].str.strip().str.upper()
    return df


def load_price_pool(price_path=PRICE_FILE):
    """The 2026 price file's full player pool: usable rows (real
    Player_ID, via load_prices()'s 'P'+zfill(6) construction -- reused,
    not reimplemented) plus placeholder-ID rows (ID literally '-',
    Player_ID left as None here -- recover_placeholder_id() fills it in
    per-player, only for players actually on the roster). Position is
    mapped from the sheet's full word to the G/F/C code used everywhere
    else in the engine."""
    prices_df, excluded = load_prices(prices_path=price_path)

    usable = prices_df[["Player_ID", "Name", "Surname", "Team", "Position", "Quotation"]].copy()
    usable["is_placeholder"] = False

    placeholder = excluded["placeholder_id"][["Name", "Surname", "Team", "Position", "Quotation"]].copy()
    placeholder["Player_ID"] = None
    placeholder["is_placeholder"] = True

    pool = pd.concat([usable, placeholder], ignore_index=True)
    pool["position"] = pool["Position"].map(POSITION_NAME_TO_CODE)

    bad_position = pool["position"].isna()
    if bad_position.any():
        raise ValueError(
            f"{price_path}: unknown Position value(s) for "
            f"{pool.loc[bad_position, ['Name', 'Surname', 'Position']].to_dict('records')} "
            f"-- known values: {sorted(POSITION_NAME_TO_CODE)}"
        )

    pool["Player_norm"] = (
        pool["Surname"].astype(str).str.strip().str.upper()
        + ", "
        + pool["Name"].astype(str).str.strip().str.upper()
    )
    return pool


def recover_placeholder_id(name, price_team_code, boxscore_players):
    """Real Player_ID for a '-'-placeholder price-file row, by matching
    name + team against RECOVERY_SEASON boxscores. Translates the price
    sheet's team code via PRICE_SHEET_TEAM_CODE_MAP (optimizer/
    coach_ranking.py, reused). Fails loudly, naming the player, if the
    team code isn't mapped or if name+team matches zero or more than one
    Player_ID."""
    boxscores_team = PRICE_SHEET_TEAM_CODE_MAP.get(price_team_code)
    if boxscores_team is None:
        raise ValueError(
            f"{MY_TEAM_CSV}: {name!r} has placeholder ID '-' in {PRICE_FILE} and "
            f"its price-sheet team code {price_team_code!r} isn't in "
            f"PRICE_SHEET_TEAM_CODE_MAP (optimizer/coach_ranking.py) -- can't "
            f"recover a real Player_ID."
        )

    name_norm = name.strip().upper()
    matches = boxscore_players[
        (boxscore_players["Player_norm"] == name_norm)
        & (boxscore_players["Team"] == boxscores_team)
    ]

    if len(matches) == 0:
        raise ValueError(
            f"{MY_TEAM_CSV}: {name!r} has placeholder ID '-' in {PRICE_FILE} and "
            f"could NOT be recovered -- zero Season {RECOVERY_SEASON} boxscores "
            f"players named {name!r} on team {boxscores_team!r} (mapped from "
            f"price-sheet code {price_team_code!r}). Check spelling, or this "
            f"player may be new to EuroLeague in 2026."
        )

    distinct_ids = matches["Player_ID"].unique()
    if len(distinct_ids) > 1:
        raise ValueError(
            f"{MY_TEAM_CSV}: {name!r} placeholder-ID recovery is ambiguous -- "
            f"{len(distinct_ids)} different Season {RECOVERY_SEASON} Player_IDs "
            f"match name+team: {list(distinct_ids)}"
        )

    return distinct_ids[0]


def resolve_player_ids(names, price_pool, boxscore_players):
    """Resolve each typed name against price_pool (the 2026 price file's
    player pool). A normal match returns its real Player_ID directly; a
    match on a placeholder-ID row triggers recover_placeholder_id(). Same
    fail-loud-on-zero-or-ambiguous contract as
    optimizer/availability.py's load_unavailable(). Returns a DataFrame
    (player_name as typed, Player_ID, position, price, Team)."""
    rows = []
    for name in names:
        name_norm = name.strip().upper()
        matches = price_pool[price_pool["Player_norm"] == name_norm]

        if len(matches) == 0:
            raise ValueError(
                f"{MY_TEAM_CSV}: {name!r} matched zero players in {PRICE_FILE}. "
                f"Check spelling/format -- name is 'LASTNAME, FIRSTNAME' (e.g. "
                f"'VEZENKOV, SASHA'), same convention as data/unavailable.csv."
            )
        if len(matches) > 1:
            raise ValueError(
                f"{MY_TEAM_CSV}: {name!r} matched {len(matches)} rows in "
                f"{PRICE_FILE}: {matches[['Name', 'Surname', 'Team']].to_dict('records')}. "
                f"Name is ambiguous -- this loader has no way to disambiguate."
            )

        row = matches.iloc[0]
        player_id = (
            recover_placeholder_id(name, row["Team"], boxscore_players)
            if row["is_placeholder"]
            else row["Player_ID"]
        )
        rows.append(
            {
                "player_name": name,
                "Player_ID": player_id,
                "position": row["position"],
                "price": row["Quotation"],
                "Team": row["Team"],
            }
        )

    return pd.DataFrame(rows)


def load_coach_pool(price_path=PRICE_FILE):
    """Coaches from the SAME price file the players come from (price_path), via
    load_prices()'s excluded['head_coaches'] -- so coach and player prices can't
    come from different snapshots. Same columns as coach_ranking.load_coach_info()
    (team_code, coach, coach_price), except team_code is the price sheet's own club
    code, NOT remapped to box-score codes: PRICE_SHEET_TEAM_CODE_MAP is derived from
    2025 box scores and lacks 2026 clubs (BJK/FBT/MIL), so load_coach_info() would
    raise on this file. Here team_code is only reported/disambiguated."""
    _, excluded = load_prices(prices_path=price_path)
    coaches = excluded["head_coaches"].copy()
    coaches["team_code"] = coaches["Team"]
    coaches["coach"] = coaches["Name"].str.strip() + " " + coaches["Surname"].str.strip()
    coaches = coaches.rename(columns={"Quotation": "coach_price"})
    return coaches[["team_code", "coach", "coach_price"]]


def resolve_coach(name, coach_info):
    """Match name against load_coach_pool()'s 'coach' column (First Last; the
    price_path file's head coaches -- same file as the players), case-insensitive,
    same fail-loud contract as resolve_player_ids."""
    name_norm = name.strip().upper()
    matches = coach_info[coach_info["coach"].str.strip().str.upper() == name_norm]

    if len(matches) == 0:
        raise ValueError(
            f"{MY_TEAM_CSV}: coach {name!r} matched zero coaches in the price "
            f"price file's head-coach list (the price_path file). Check "
            f"spelling -- expected format 'First Last' (e.g. 'Sergio Scariolo')."
        )
    if matches["coach"].nunique() > 1 or matches["team_code"].nunique() > 1:
        raise ValueError(
            f"{MY_TEAM_CSV}: coach {name!r} matched {len(matches)} ambiguous rows: "
            f"{matches[['coach', 'team_code']].to_dict('records')}"
        )
    return matches.iloc[0]


def load_my_team_state(state_path=MY_TEAM_STATE_CSV):
    df = pd.read_csv(state_path, comment="#")
    state = dict(zip(df["field"], df["value"]))
    required = {"remaining_budget", "trades_used"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"{state_path} is missing required field(s): {sorted(missing)}")
    return {
        "remaining_budget": float(state["remaining_budget"]),
        "trades_used": int(state["trades_used"]),
    }


def validate_roster(players, captain_role):
    """Fails loudly on any roster-shape violation. players must already
    have 'role' and 'position' columns attached; captain_role is the role
    of whichever row has captain=True (or None if zero captains)."""
    if len(players) != 10:
        raise ValueError(f"Expected 10 players (excluding coach), found {len(players)}")

    role_counts = players["role"].value_counts().to_dict()
    if role_counts != ROLE_REQUIREMENTS:
        raise ValueError(
            f"Role counts {role_counts} don't match the required "
            f"{ROLE_REQUIREMENTS} (5 starter, 1 6th_man, 4 bench)"
        )

    position_counts = players["position"].value_counts().to_dict()
    if position_counts != POSITION_REQUIREMENTS:
        raise ValueError(
            f"Position counts {position_counts} don't match the required "
            f"{POSITION_REQUIREMENTS} (4G/4F/2C)"
        )

    n_captains = players["captain"].apply(_is_captain).sum()
    if n_captains != 1:
        raise ValueError(f"Expected exactly one captain, found {n_captains}")

    if captain_role != "starter":
        raise ValueError(
            f"The captain must be a starter (role={captain_role!r}) -- same "
            f"ASSUMPTION optimizer/pick_lineup_ilp.py's own docstring flags as "
            f"unverified against the live app; verify there if this ever needs "
            f"to change."
        )


def load_my_team(
    csv_path=MY_TEAM_CSV,
    state_path=MY_TEAM_STATE_CSV,
    price_path=PRICE_FILE,
    recovery_season=RECOVERY_SEASON,
    db_path=DB_PATH,
):
    """The reusable entry point: load, resolve, and validate the current
    team. Returns {'players': DataFrame, 'coach': dict, 'state': dict,
    'season': SEASON, 'price_round': PRICE_ROUND}."""
    roster_raw = load_roster_csv(csv_path)

    coach_rows = roster_raw[roster_raw["role"] == "coach"]
    if len(coach_rows) != 1:
        raise ValueError(f"{csv_path}: expected exactly 1 coach row, found {len(coach_rows)}")
    players_raw = roster_raw[roster_raw["role"] != "coach"].copy()

    price_pool = load_price_pool(price_path)
    boxscore_players = load_boxscore_players(recovery_season, db_path)
    resolved = resolve_player_ids(players_raw["player_name"].tolist(), price_pool, boxscore_players)

    players = players_raw.merge(resolved, on="player_name", how="left")

    coach_info = load_coach_pool(price_path)  # same file as the players
    coach_match = resolve_coach(coach_rows.iloc[0]["player_name"], coach_info)

    captain_rows = players[players["captain"].apply(_is_captain)]
    captain_role = captain_rows.iloc[0]["role"] if len(captain_rows) == 1 else None
    validate_roster(players, captain_role)

    coach = {
        "name": coach_match["coach"],
        "team_code": coach_match["team_code"],
        "price": coach_match["coach_price"],
    }
    state = load_my_team_state(state_path)

    return {
        "players": players,
        "coach": coach,
        "state": state,
        "season": SEASON,
        "price_round": PRICE_ROUND,
    }


def print_report(loaded):
    players = loaded["players"].copy()
    role_order = {"starter": 0, "6th_man": 1, "bench": 2}
    players = players.sort_values(
        by="role", key=lambda col: col.map(role_order)
    )
    players["captain"] = players["captain"].apply(lambda v: "(C)" if _is_captain(v) else "")

    print(f"Season {loaded['season']}, prices from Round {loaded['price_round']} ({PRICE_FILE})")

    print(f"\nCoach: {loaded['coach']['name']} ({loaded['coach']['team_code']}) "
          f"-- price {loaded['coach']['price']}")

    columns = ["player_name", "role", "captain", "position", "Team", "price"]
    print("\nRoster:")
    print(players[columns].to_string(index=False))

    total_price = players["price"].sum() + loaded["coach"]["price"]
    print(f"\nRoster + coach total price: {total_price:.1f}")
    print(f"Remaining budget (data/my_team_state.csv): {loaded['state']['remaining_budget']}")
    print(f"Trades used (data/my_team_state.csv): {loaded['state']['trades_used']}")


if __name__ == "__main__":
    loaded = load_my_team()
    print_report(loaded)
