"""Extends optimizer/pick_team_ilp.py with the actual scoring lineup, per
data requirements doc Sections 1-2: a 10-man roster (4G/4F/2C, budget<=100,
max 6/club, unchanged from pick_team_ilp.py), from which a starting five +
6th man score at 100% and the remaining 4 bench players score at 50%. The
starting five must match one legal formation (G-F-C counts); the 6th man
has no position restriction.

This is one combined MILP, not two separate stages -- the roster and the
lineup are optimized jointly against the weighted objective below, so the
solver can trade off "who's on the roster" against "who starts" together.

Slice B adds the captain: exactly one captain, chosen from the starting
five, whose score is multiplied by the season's captain_multiplier
(scoring/season_rules.py, via get_rules -- never hardcoded).

ASSUMPTION TO VERIFY AGAINST THE LIVE APP: the rulebook doesn't explicitly
state whether a bench player can be captained. This code assumes the
captain must be a starter. If the live app allows captaining a bench or
6th-man player, this constraint needs loosening.

Reuses ROUND_GAMECODES, POSITION_REQUIREMENTS, attach_positions, and
get_round_season (optimizer/pick_team.py), BUDGET, MAX_PER_CLUB
(optimizer/pick_team_ilp.py), train_and_predict (prediction/model_v3.py),
get_rules (scoring/season_rules.py), load_unavailable
(optimizer/availability.py), and load_prices
(data_collection/load_prices.py) -- nothing here is recomputed. Positions
are real (see pick_team.py's docstring); prices are now real quotations
from the price sheet, joined on Player_ID (no longer the dummy_price
placeholder); club is real data -- same as pick_team_ilp.py.

Before the roster/lineup MILP runs, players listed in data/unavailable.csv
(hand-edited, out/doubtful for the upcoming round) are dropped from
df_round entirely -- they never enter player_vars, so the solver has no way
to select them at any weight, unlike zero-scoring them which would still
let them occupy a roster slot for free. Candidates with no matching price
are dropped from the pool the same way, for the same reason.

NOTE ON PRICE COVERAGE: load_prices' own coverage report (run it directly)
shows only 1 of 355 real Season=2025 players lack a price. The much larger
exclusion count seen when *this* file runs (dozens, not one) is a separate,
expected effect: the candidate pool here is the 2023 validation round, and
the price sheet is a current snapshot -- most 2023 players have since
retired or left EuroLeague and were never going to be in a current price
list. This gap should shrink to near-zero once the optimizer runs against
a real current round (the actual 2026-27 live target), which is the only
setting where "2023 roster, current prices" wouldn't be a mismatch by
construction.

BUG SURFACED BY THIS CHANGE, FIXED HERE: ROUND_GAMECODES (253-261) was
calibrated against the 2023 test set only. Once data_collection/load_prices.py
added 2025 boxscores to the same database, that Gamecode range started
matching games in *both* seasons (schedules restart game numbering each
season), which broke get_round_season's "exactly one season per round"
assertion. df_round now explicitly filters to Season 2023 to restore the
original, intended round. optimizer/pick_team.py and pick_team_ilp.py have
the same latent bug -- left alone here since fixing them isn't this file's
job, but they'll raise the same AssertionError if run as-is.
"""

import pulp

from data_collection.load_prices import load_prices
from optimizer.availability import load_unavailable
from optimizer.pick_team import (
    POSITION_REQUIREMENTS,
    ROUND_GAMECODES,
    attach_positions,
    get_round_season,
)
from optimizer.pick_team_ilp import BUDGET, MAX_PER_CLUB
from prediction.model_v3 import train_and_predict
from scoring.season_rules import get_rules

SCORING_WEIGHT = 1.0
BENCH_WEIGHT = 0.5

FORMATIONS = {
    "2-2-1": {"G": 2, "F": 2, "C": 1},
    "1-2-2": {"G": 1, "F": 2, "C": 2},
    "2-1-2": {"G": 2, "F": 1, "C": 2},
    "1-3-1": {"G": 1, "F": 3, "C": 1},
    "3-1-1": {"G": 3, "F": 1, "C": 1},
}

if __name__ == "__main__":
    model, feature_cols, train, test = train_and_predict()

    # Season=2023 pins this to the round ROUND_GAMECODES was calibrated
    # against -- see the module docstring's "BUG SURFACED" note.
    df_round = test[test["Gamecode"].isin(ROUND_GAMECODES) & (test["Season"] == 2023)].copy()
    df_round = attach_positions(df_round)

    # --- exclude out/doubtful players (data/unavailable.csv) from the candidate
    # --- pool entirely, rather than zero-scoring them, so the solver can't pick
    # --- them at any weight ---
    unavailable = load_unavailable()
    excluded_ids = {entry["player_id"] for entry in unavailable}
    excluded_mask = df_round["Player_ID"].str.strip().isin(excluded_ids)
    excluded_names = df_round.loc[excluded_mask, "Player"].unique().tolist()
    df_round = df_round[~excluded_mask].copy()

    print(f"Excluded {len(excluded_names)} unavailable player(s) from the candidate pool:")
    for name in excluded_names:
        print(f"  {name}")

    # --- join real quotations onto the candidate pool (Player_ID needs
    # --- .str.strip() -- boxscores.Player_ID is fixed-width, padded with
    # --- trailing spaces, e.g. 'P005985   ') ---
    prices_df, _ = load_prices()
    price_lookup = dict(zip(prices_df["Player_ID"], prices_df["Quotation"]))
    df_round["price"] = df_round["Player_ID"].str.strip().map(price_lookup)

    # --- exclude candidates with no matching price from the pool entirely,
    # --- same treatment as the unavailable-players exclusion above, rather
    # --- than letting a missing price default to 0/NaN and distort the
    # --- budget constraint ---
    no_price_mask = df_round["price"].isna()
    no_price_names = df_round.loc[no_price_mask, "Player"].unique().tolist()
    df_round = df_round[~no_price_mask].copy()

    print(f"Excluded {len(no_price_names)} player(s) with no matching price from the candidate pool:")
    for name in no_price_names:
        print(f"  {name}")

    season = get_round_season(df_round)
    captain_multiplier = get_rules(season)["captain_multiplier"]

    prob = pulp.LpProblem("euroleague_fantasy_lineup", pulp.LpMaximize)

    player_vars = {
        idx: pulp.LpVariable(f"roster_{idx}", cat="Binary") for idx in df_round.index
    }
    starter_vars = {
        idx: pulp.LpVariable(f"starter_{idx}", cat="Binary") for idx in df_round.index
    }
    sixth_man_vars = {
        idx: pulp.LpVariable(f"sixth_{idx}", cat="Binary") for idx in df_round.index
    }
    formation_vars = {f: pulp.LpVariable(f"formation_{f}", cat="Binary") for f in FORMATIONS}
    captain_vars = {
        idx: pulp.LpVariable(f"captain_{idx}", cat="Binary") for idx in df_round.index
    }

    # --- roster constraints, unchanged from pick_team_ilp.py ---
    for position, count in POSITION_REQUIREMENTS.items():
        prob += (
            pulp.lpSum(
                player_vars[idx]
                for idx in df_round.index
                if df_round.loc[idx, "position"] == position
            )
            == count
        )

    prob += (
        pulp.lpSum(player_vars[idx] * df_round.loc[idx, "price"] for idx in df_round.index)
        <= BUDGET
    )

    for club in df_round["Team"].unique():
        prob += (
            pulp.lpSum(
                player_vars[idx] for idx in df_round.index if df_round.loc[idx, "Team"] == club
            )
            <= MAX_PER_CLUB
        )

    # --- scoring-group membership: only rostered players can start or be 6th man ---
    for idx in df_round.index:
        prob += starter_vars[idx] <= player_vars[idx]
        prob += sixth_man_vars[idx] <= player_vars[idx]
        prob += starter_vars[idx] + sixth_man_vars[idx] <= 1

    prob += pulp.lpSum(starter_vars.values()) == 5
    prob += pulp.lpSum(sixth_man_vars.values()) == 1

    # --- captain: exactly one, must be a starter (ASSUMPTION, see module docstring --
    # --- verify against the live app whether a bench/6th-man player can be captained) ---
    for idx in df_round.index:
        prob += captain_vars[idx] <= starter_vars[idx]
    prob += pulp.lpSum(captain_vars.values()) == 1

    # --- exactly one formation, and starters' position mix must match it ---
    prob += pulp.lpSum(formation_vars.values()) == 1

    for position in POSITION_REQUIREMENTS:
        prob += (
            pulp.lpSum(
                starter_vars[idx]
                for idx in df_round.index
                if df_round.loc[idx, "position"] == position
            )
            == pulp.lpSum(formation_vars[f] * FORMATIONS[f][position] for f in FORMATIONS)
        )

    # --- objective: 100% scoring group (starters + 6th man) + 50% bench, plus the
    # --- captain's *extra* (multiplier - 1) share on top of their already-counted 100% ---
    scoring_group_score = pulp.lpSum(
        df_round.loc[idx, "predicted_score"] * (starter_vars[idx] + sixth_man_vars[idx])
        for idx in df_round.index
    )
    bench_score = pulp.lpSum(
        df_round.loc[idx, "predicted_score"]
        * (player_vars[idx] - starter_vars[idx] - sixth_man_vars[idx])
        for idx in df_round.index
    )
    captain_bonus = pulp.lpSum(
        df_round.loc[idx, "predicted_score"] * captain_vars[idx] for idx in df_round.index
    )
    prob += (
        SCORING_WEIGHT * scoring_group_score
        + BENCH_WEIGHT * bench_score
        + (captain_multiplier - 1) * captain_bonus
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    starters_idx = [idx for idx in df_round.index if starter_vars[idx].value() == 1]
    sixth_idx = [idx for idx in df_round.index if sixth_man_vars[idx].value() == 1]
    bench_idx = [
        idx
        for idx in df_round.index
        if player_vars[idx].value() == 1 and idx not in starters_idx and idx not in sixth_idx
    ]
    active_formation = next(f for f in FORMATIONS if formation_vars[f].value() == 1)
    captain_idx = next(idx for idx in df_round.index if captain_vars[idx].value() == 1)
    roster_idx = starters_idx + sixth_idx + bench_idx

    # --- fail loudly rather than silently defaulting, if the solver somehow
    # --- selected a candidate with no price (shouldn't happen -- they were
    # --- dropped from the pool above -- so this guards against that
    # --- exclusion logic silently breaking in the future) ---
    missing_price = df_round.loc[roster_idx, "price"].isna()
    if missing_price.any():
        raise ValueError(
            f"Solver selected candidate(s) with no price, which should be "
            f"impossible -- the pool exclusion above must have a bug: "
            f"{df_round.loc[roster_idx][missing_price]['Player'].tolist()}"
        )

    display = df_round.copy()
    display["Player"] = display["Player"] + display.index.to_series().apply(
        lambda idx: " (C)" if idx == captain_idx else ""
    )
    columns = ["Player", "position", "price", "predicted_score"]

    print(f"Solver status: {status}")
    print(f"Formation: {active_formation}")
    print(f"Season: {season}, captain_multiplier: {captain_multiplier}")

    print("\nStarting five:")
    print(display.loc[starters_idx, columns].to_string(index=False))

    print("\n6th man:")
    print(display.loc[sixth_idx, columns].to_string(index=False))

    print("\nBench (50%):")
    print(display.loc[bench_idx, columns].to_string(index=False))

    captain_score = df_round.loc[captain_idx, "predicted_score"]
    weighted_total = (
        df_round.loc[starters_idx + sixth_idx, "predicted_score"].sum() * SCORING_WEIGHT
        + df_round.loc[bench_idx, "predicted_score"].sum() * BENCH_WEIGHT
        + (captain_multiplier - 1) * captain_score
    )
    total_quotation = df_round.loc[roster_idx, "price"].sum()
    print(f"\nTotal quotation used: {total_quotation:.1f} / {BUDGET}")
    print(f"Weighted total predicted score: {weighted_total:.2f}")
