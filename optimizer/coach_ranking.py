"""Crude team-strength ranking to shortlist coaches for a target round.

Not a fantasy-points model -- this ranks TEAMS by recent real-game point
margin (Points scored - Points conceded), as a rough proxy for "which
teams are playing well right now", then surfaces the top 5 teams' coaches
for a target round. No DB writes -- read-only, report only.

Reuses match_opponents (features/matchup.py -- extracted from
add_team_rolling_defense specifically so this file's real-points margin
and the existing fantasy-conceded feature share the same opponent-matching
self-merge instead of two separate reimplementations), load_prices
(data_collection/load_prices.py -- its excluded['head_coaches'] already
holds every team's coach, parsed out of the price sheet; not reparsed
here), and ROUND_GAMECODES_2025 (optimizer/pick_lineup_ilp.py -- the same
live target round the optimizer itself uses, not a new one picked here) --
nothing here is recomputed.

LEAKAGE-FREE ROLLING MARGIN: for each team-game, margin = own Points -
opponent's Points (from match_opponents). rolling_5_margin is shift(1) +
rolling(5, min_periods=1).mean(), keyed on (Season, Team) so a team's
window never crosses a season boundary (Gamecode resets each season --
see features/rolling_form.py for the earlier fix for the same bug class).
A target round's OWN game is never in its own rolling_5_margin -- shift(1)
guarantees that -- so "recent team strength entering the round" is read
directly off the target round's row, no separate blinding step needed
(unlike backtest/simulate_round.py's engine, nothing here is TRAINED on
this data, so there's no analogous risk of the round's own result leaking
into a fitted model).

TEAM CODE MISMATCH (the one real gotcha here): the price sheet
(data/prices/players_stats.xlsx) uses a DIFFERENT team-code convention
than boxscores.Team -- e.g. coach Sergio Scariolo is listed under 'RMB',
not Real Madrid's boxscores code 'MAD'. This isn't season drift --
boxscores' own code scheme is identical across Season 2023 and 2025 -- the
price sheet is simply a different vendor's codes. PRICE_SHEET_TEAM_CODE_MAP
below was derived FROM DATA, not guessed: every price-sheet player's
Player_ID was cross-referenced against their real Season=2025
boxscores.Team, giving a near-unanimous majority per team (16-21 agreeing
players each; the handful of single-player disagreements are mid-season
transfers, not code ambiguity). Re-derive by rerunning that cross-reference
if the price sheet is ever refreshed.

RANKING ORDER (a judgment call on an ambiguous instruction -- flag if a
different priority was intended): teams are ranked by rolling_5_margin
descending, since "rank the teams... by their recent-margin" names that as
the primary criterion. Home-game status is used only as a TIEBREAKER
(home before away) among otherwise-equal-margin teams, per "prioritize
home games" -- not as a stronger boost that could rank a mediocre home
team above a clearly stronger road team.
"""

import sqlite3

import pandas as pd

from data_collection.load_prices import load_prices
from features.matchup import match_opponents
from optimizer.pick_lineup_ilp import ROUND_GAMECODES_2025

DB_PATH = "data/euroleague.db"
SEASON = 2025
SHORTLIST_SIZE = 5

# Derived from load_prices() player rows' Team vs. their real Season=2025
# boxscores.Team (see module docstring) -- price-sheet code -> boxscores code.
PRICE_SHEET_TEAM_CODE_MAP = {
    "ASM": "MCO",
    "ASV": "ASV",
    "BAR": "BAR",
    "BAY": "MUN",
    "CZV": "RED",
    "DUB": "DUB",
    "EA7": "MIL",
    "EFS": "IST",
    "FBB": "ULK",
    "HTA": "HTA",
    "KBA": "BAS",
    "MTA": "TEL",
    "OLY": "OLY",
    "PAO": "PAN",
    "PAR": "PAR",
    "PBB": "PRS",
    "RMB": "MAD",
    "VBC": "PAM",
    "VIR": "VIR",
    "ZAL": "ZAL",
}


def load_season_boxscores():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM boxscores WHERE Season = ? AND Player_ID NOT IN ('Team', 'Total')",
        conn,
        params=[SEASON],
    )
    conn.close()
    return df


def compute_rolling_margin(boxscores):
    """Leakage-free rolling_5_margin per team-game -- see module docstring."""
    matchup = match_opponents(boxscores, "Points")
    matchup = matchup.rename(columns={"Points": "points_for", "Points_opp": "points_against"})
    matchup["margin"] = matchup["points_for"] - matchup["points_against"]

    matchup = matchup.sort_values(["Team", "Season", "Gamecode"]).reset_index(drop=True)
    by_team = matchup.groupby(["Season", "Team"])["margin"]
    matchup["rolling_5_margin"] = by_team.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    return matchup


def load_home_flags(boxscores):
    """Per (Season, Gamecode, Team), whether that team was the home side.
    Home is constant across every player row for a team in a game, so any
    aggregation (here, first()) recovers the team-level flag."""
    return boxscores.groupby(["Season", "Gamecode", "Team"])["Home"].first().reset_index()


def load_coaches():
    """boxscores-code Team -> coach full name. load_prices()'s
    excluded['head_coaches'] already parsed these out of the price sheet;
    only remapped here through PRICE_SHEET_TEAM_CODE_MAP onto boxscores'
    team codes, never reparsed."""
    _, excluded = load_prices()
    coaches = excluded["head_coaches"].copy()
    coaches["team_code"] = coaches["Team"].map(PRICE_SHEET_TEAM_CODE_MAP)

    unmapped = coaches["team_code"].isna()
    if unmapped.any():
        raise ValueError(
            f"PRICE_SHEET_TEAM_CODE_MAP is missing price-sheet code(s): "
            f"{coaches.loc[unmapped, 'Team'].tolist()}"
        )

    coach_name = coaches["Name"].str.strip() + " " + coaches["Surname"].str.strip()
    return dict(zip(coaches["team_code"], coach_name))


if __name__ == "__main__":
    boxscores = load_season_boxscores()

    matchup = compute_rolling_margin(boxscores)
    home_flags = load_home_flags(boxscores)
    coach_lookup = load_coaches()

    target_round = matchup[matchup["Gamecode"].isin(ROUND_GAMECODES_2025)].copy()
    target_round = target_round.merge(home_flags, on=["Season", "Gamecode", "Team"], how="left")

    # --- fail loudly rather than silently ranking with a missing value ---
    missing_margin = target_round["rolling_5_margin"].isna()
    if missing_margin.any():
        raise ValueError(
            f"Team(s) with no rolling_5_margin entering the target round -- "
            f"cold start (fewer than 1 prior Season {SEASON} game): "
            f"{target_round.loc[missing_margin, 'Team'].tolist()}"
        )

    target_round["coach"] = target_round["Team"].map(coach_lookup)
    missing_coach = target_round["coach"].isna()
    if missing_coach.any():
        raise ValueError(
            f"Team(s) with no coach found, even after PRICE_SHEET_TEAM_CODE_MAP: "
            f"{target_round.loc[missing_coach, 'Team'].tolist()}"
        )

    # --- rank by recent-margin (primary), home status as tiebreaker only
    # --- (see module docstring's RANKING ORDER note) ---
    target_round = target_round.sort_values(
        ["rolling_5_margin", "Home"], ascending=[False, False]
    ).reset_index(drop=True)

    print(
        f"Target round: Season {SEASON}, Gamecode "
        f"{min(ROUND_GAMECODES_2025)}-{max(ROUND_GAMECODES_2025)}"
    )

    display = target_round.copy()
    display["Home"] = display["Home"].map({1: "Yes", 0: "No"})
    display = display.rename(
        columns={"rolling_5_margin": "recent_avg_margin", "Team_opp": "opponent"}
    )
    columns = ["Team", "coach", "recent_avg_margin", "opponent", "Home"]

    print(f"\nAll {len(display)} teams playing this round, ranked by recent-margin (home as tiebreaker):")
    print(display[columns].to_string(index=False))

    shortlist = display.head(SHORTLIST_SIZE)
    print(f"\nTop {SHORTLIST_SIZE} coach shortlist:")
    print(shortlist[columns].to_string(index=False))
