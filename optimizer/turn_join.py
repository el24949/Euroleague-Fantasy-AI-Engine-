"""Join each candidate player in the target round's pool to their team's
turn (T1/T2/RESCHEDULED), using features/turn_tags.py's turn lookup.

Slice-only script -- no optimizer objective change yet. Builds the SAME
candidate pool optimizer/pick_lineup_ilp.py builds up through
attach_positions (train_and_predict -> filter to ROUND_GAMECODES_2025 &
Season==2025 -> attach_positions), reusing that logic rather than
reimplementing it, then joins turn onto it via
features.turn_tags.build_turn_lookup(season).

WHY (Season, Gamecode) IS THE JOIN KEY: build_turn_lookup returns
{(season, game): turn}, keyed on the schedule's "game" field -- the same
field boxscores.Gamecode is sourced from (confirmed in
data_collection/fetch_season.py: gamecodes = round_games["game"]...; the
same (Season, Gamecode) key is already relied on by
optimizer/coach_ranking.py's rolling-margin join). Each candidate row's own
Gamecode IS the game their team plays this round (ROUND_GAMECODES_2025
already filters df_round to exactly those games), so a player's turn is
just the turn of the game their row is already tagged with -- no separate
team->game lookup is needed.

RESCHEDULED / MISSING HANDLING: build_turn_lookup labels a game
RESCHEDULED (not T1 or T2) when a round has more than 2 distinct dates and
a game doesn't fall on either of the two core dates (see turn_tags.py's
module docstring for why this isn't a real 3rd turn). A (season, Gamecode)
key can also simply be absent from the lookup (build_turn_lookup only
covers regular-season games). Both cases are flagged here rather than
crashing, since a candidate in either state still needs *some* scoring
treatment -- deciding what that is belongs to a later slice.
"""

import pandas as pd

from features.turn_tags import build_turn_lookup
from optimizer.pick_team import ROUND_GAMECODES_2025, attach_positions, get_round_season
from prediction.model_v3 import train_and_predict


def attach_turns(df_round, turn_lookup):
    """Add a 'turn' column: T1/T2/RESCHEDULED from the lookup, or
    NO_TURN_FOUND if (Season, Gamecode) has no entry at all."""
    df_round = df_round.copy()
    keys = list(zip(df_round["Season"], df_round["Gamecode"]))
    df_round["turn"] = [turn_lookup.get(key, "NO_TURN_FOUND") for key in keys]
    return df_round


if __name__ == "__main__":
    model, feature_cols, train, test = train_and_predict()

    df_round = test[
        test["Gamecode"].isin(ROUND_GAMECODES_2025) & (test["Season"] == 2025)
    ].copy()
    df_round = attach_positions(df_round)

    season = get_round_season(df_round)
    turn_lookup = build_turn_lookup(season)
    df_round = attach_turns(df_round, turn_lookup)

    flagged = df_round[df_round["turn"].isin(["RESCHEDULED", "NO_TURN_FOUND"])]
    if len(flagged):
        print(f"Flagged {len(flagged)} candidate row(s) with no clean T1/T2 turn:")
        print(
            flagged[["Player", "Team", "Gamecode", "turn"]]
            .drop_duplicates()
            .to_string(index=False)
        )
    else:
        print("No flagged candidates -- every game in this round joined cleanly to T1/T2.")

    print("\nSample of candidates (Player, Team, position, turn):")
    sample = df_round[["Player", "Team", "position", "turn"]].drop_duplicates(subset=["Player"])
    sample_size = min(15, len(sample))
    print(
        sample.sample(sample_size, random_state=0)
        .sort_values(["turn", "Team"])
        .to_string(index=False)
    )

    print("\nCandidate count by turn:")
    print(df_round["turn"].value_counts().to_string())
