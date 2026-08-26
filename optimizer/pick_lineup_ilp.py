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

Reuses POSITION_REQUIREMENTS, attach_positions, and get_round_season
(optimizer/pick_team.py), BUDGET, MAX_PER_CLUB (optimizer/pick_team_ilp.py),
train_and_predict (prediction/model_v3.py), get_rules
(scoring/season_rules.py), load_unavailable (optimizer/availability.py),
and load_prices (data_collection/load_prices.py) -- nothing here is
recomputed. Positions are real (see pick_team.py's docstring); prices are
real quotations from the price sheet, joined on Player_ID; club is real
data -- same as pick_team_ilp.py.

The MILP itself lives in solve_lineup(df_round, score_col, ...) so
backtest/simulate_round.py can reuse the exact same roster+lineup
constraints (4G/4F/2C, budget, max 6/club, formations, 6th man, captain)
while optimizing a *different* per-player score column -- e.g. a naive
season-to-date average or a random baseline -- without duplicating this
MILP. score_col is deliberately a parameter rather than a hardcoded
"predicted_score" for exactly that reason.

RISK-AWARE OBJECTIVE: the engine solves against objective_score =
predicted_score - RISK_LAMBDA * consistency_5, not raw predicted_score.
consistency_5 (a player's rolling std of fantasy score -- how volatile
they've been, not how good) is already a leakage-free feature computed by
model_v3.train_and_predict() via add_rolling_form (features/rolling_form.py)
-- it survives untouched through attach_positions() and the price join
below (neither drops columns), so it's just read off df_round, never
recomputed or rejoined. RISK_LAMBDA=0.25 was tuned by sweeping
backtest/simulate_many.py's LAMBDA values across Season 2025 rounds 12-34:
avg actual points rose from 107.54 (LAMBDA=0, i.e. today's raw-predicted
engine) to 112.39, and rounds beating the naive baseline rose from 10/23
to 14/23. This is one season's worth of evidence -- re-validate against
more seasons before trusting the exact value 0.25 long-term.

Before the roster/lineup MILP runs, players listed in data/unavailable.csv
(hand-edited, out/doubtful for the upcoming round) are dropped from
df_round entirely -- they never enter player_vars, so the solver has no way
to select them at any weight, unlike zero-scoring them which would still
let them occupy a roster slot for free. Candidates with no matching price
are dropped from the pool the same way, for the same reason.

RUNS FULLY ON SEASON 2025 (previously it validated against a 2023 round
while joining current-season prices, a mismatch that excluded dozens of
players who'd since retired or left EuroLeague -- see git history for that
version). model_v3.train_and_predict() now trains and tests entirely within
Season 2025 (prediction/model_v3.py), so its `test` DataFrame is already
100% Season 2025 by construction. ROUND_GAMECODES_2025 below picks one real
round from within that test range (Gamecode >= model_v3.CUTOFF_GAMECODE),
and df_round still filters on Season == 2025 explicitly rather than relying
on that invariant -- if train_and_predict's split logic ever changes to mix
seasons again, this file fails loudly (via get_round_season's assertion)
instead of silently picking a blended-season "round" again. Since the round
is now real 2025 games, price coverage against it should be close to
load_prices' own report (currently 354/355 real Season=2025 players
priced).
"""

import pulp

from data_collection.load_prices import load_prices
from optimizer.availability import load_unavailable
from optimizer.pick_team import POSITION_REQUIREMENTS, attach_positions, get_round_season
from optimizer.pick_team_ilp import BUDGET, MAX_PER_CLUB
from prediction.model_v3 import train_and_predict
from scoring.season_rules import get_rules

SCORING_WEIGHT = 1.0
BENCH_WEIGHT = 0.5

# Tuned via backtest/simulate_many.py's LAMBDA sweep on Season 2025 rounds
# 12-34: avg actual points 107.54 -> 112.39, rounds beating naive 10/23 ->
# 14/23 at LAMBDA=0.25. One season of evidence -- re-validate with more
# seasons before treating 0.25 as settled.
RISK_LAMBDA = 0.25

# One real, complete Season 2025 round: 10 consecutive Gamecodes, 20 distinct
# teams with no repeats (verified against boxscores), fully inside
# model_v3's test range (Gamecode >= CUTOFF_GAMECODE = 304). Unlike the old
# ROUND_GAMECODES (optimizer/pick_team.py, calibrated to the 2023 test set's
# 9-games/18-teams round shape), 2025 has 20 teams, so a round is 10 games.
ROUND_GAMECODES_2025 = range(311, 321)

FORMATIONS = {
    "2-2-1": {"G": 2, "F": 2, "C": 1},
    "1-2-2": {"G": 1, "F": 2, "C": 2},
    "2-1-2": {"G": 2, "F": 1, "C": 2},
    "1-3-1": {"G": 1, "F": 3, "C": 1},
    "3-1-1": {"G": 3, "F": 1, "C": 1},
}


def solve_lineup(
    df_round,
    score_col,
    captain_multiplier,
    budget=BUDGET,
    max_per_club=MAX_PER_CLUB,
    scoring_weight=SCORING_WEIGHT,
    bench_weight=BENCH_WEIGHT,
):
    """Solve the roster+lineup MILP over df_round, maximizing score_col.

    df_round must already have 'position', 'price', 'Team', and score_col
    columns, and a unique index (used as the player identifier throughout).
    Returns a dict of index lists/labels describing the selection -- never
    a printed report, so callers (this file's __main__ block,
    backtest/simulate_round.py) can format the result however they need.
    """
    prob = pulp.LpProblem("euroleague_fantasy_lineup", pulp.LpMaximize)

    player_vars = {idx: pulp.LpVariable(f"roster_{idx}", cat="Binary") for idx in df_round.index}
    starter_vars = {idx: pulp.LpVariable(f"starter_{idx}", cat="Binary") for idx in df_round.index}
    sixth_man_vars = {
        idx: pulp.LpVariable(f"sixth_{idx}", cat="Binary") for idx in df_round.index
    }
    formation_vars = {f: pulp.LpVariable(f"formation_{f}", cat="Binary") for f in FORMATIONS}
    captain_vars = {idx: pulp.LpVariable(f"captain_{idx}", cat="Binary") for idx in df_round.index}

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
        <= budget
    )

    for club in df_round["Team"].unique():
        prob += (
            pulp.lpSum(
                player_vars[idx] for idx in df_round.index if df_round.loc[idx, "Team"] == club
            )
            <= max_per_club
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
        df_round.loc[idx, score_col] * (starter_vars[idx] + sixth_man_vars[idx])
        for idx in df_round.index
    )
    bench_score = pulp.lpSum(
        df_round.loc[idx, score_col]
        * (player_vars[idx] - starter_vars[idx] - sixth_man_vars[idx])
        for idx in df_round.index
    )
    captain_bonus = pulp.lpSum(
        df_round.loc[idx, score_col] * captain_vars[idx] for idx in df_round.index
    )
    prob += (
        scoring_weight * scoring_group_score
        + bench_weight * bench_score
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

    return {
        "status": status,
        "formation": active_formation,
        "starters_idx": starters_idx,
        "sixth_idx": sixth_idx,
        "bench_idx": bench_idx,
        "captain_idx": captain_idx,
        "roster_idx": starters_idx + sixth_idx + bench_idx,
    }


if __name__ == "__main__":
    model, feature_cols, train, test = train_and_predict()

    # test is already 100% Season 2025 (model_v3.train_and_predict asserts
    # this), but the Season == 2025 filter is kept explicit rather than
    # relying on that invariant -- see the module docstring.
    df_round = test[
        test["Gamecode"].isin(ROUND_GAMECODES_2025) & (test["Season"] == 2025)
    ].copy()
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

    # --- risk-aware objective: predicted_score - RISK_LAMBDA * consistency_5.
    # --- consistency_5 is guaranteed non-null pool-wide by model_v3's own
    # --- dropna(subset=feature_cols) (prediction/model_v3.py), the same
    # --- upstream guarantee attach_positions() gives 'position' -- but
    # --- checked explicitly here, before it's used as an LP coefficient,
    # --- rather than letting a NaN silently corrupt the objective or crash
    # --- deep inside solve_lineup with a confusing PuLP error ---
    missing_consistency = df_round["consistency_5"].isna()
    if missing_consistency.any():
        raise ValueError(
            f"{missing_consistency.sum()} candidate(s) have no consistency_5, "
            f"which should be impossible -- model_v3.train_and_predict()'s own "
            f"dropna(subset=feature_cols) must have a bug: "
            f"{df_round.loc[missing_consistency, 'Player'].tolist()}"
        )
    df_round["objective_score"] = (
        df_round["predicted_score"] - RISK_LAMBDA * df_round["consistency_5"]
    )

    result = solve_lineup(df_round, "objective_score", captain_multiplier)
    status = result["status"]
    active_formation = result["formation"]
    starters_idx = result["starters_idx"]
    sixth_idx = result["sixth_idx"]
    bench_idx = result["bench_idx"]
    captain_idx = result["captain_idx"]
    roster_idx = result["roster_idx"]

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

    # --- same guarantee for position: attach_positions() already raises for
    # --- *any* candidate missing a position before the pool is even built
    # --- (see optimizer/pick_team.py), so this can only fire if that
    # --- upstream guarantee is ever weakened ---
    missing_position = df_round.loc[roster_idx, "position"].isna()
    if missing_position.any():
        raise ValueError(
            f"Solver selected candidate(s) with no position, which should be "
            f"impossible -- attach_positions' pool-wide check must have a bug: "
            f"{df_round.loc[roster_idx][missing_position]['Player'].tolist()}"
        )

    display = df_round.copy()
    display["Player"] = display["Player"] + display.index.to_series().apply(
        lambda idx: " (C)" if idx == captain_idx else ""
    )
    # predicted_score, consistency_5, and objective_score shown side by side
    # so the risk penalty's effect on picks is visible, not just its result
    columns = ["Player", "position", "price", "predicted_score", "consistency_5", "objective_score"]

    print(f"Solver status: {status}")
    print(f"Formation: {active_formation}")
    print(f"Season: {season}, captain_multiplier: {captain_multiplier}")
    print(f"RISK_LAMBDA: {RISK_LAMBDA}")

    print("\nStarting five:")
    print(display.loc[starters_idx, columns].to_string(index=False))

    print("\n6th man:")
    print(display.loc[sixth_idx, columns].to_string(index=False))

    print("\nBench (50%):")
    print(display.loc[bench_idx, columns].to_string(index=False))

    captain_predicted = df_round.loc[captain_idx, "predicted_score"]
    weighted_predicted_total = (
        df_round.loc[starters_idx + sixth_idx, "predicted_score"].sum() * SCORING_WEIGHT
        + df_round.loc[bench_idx, "predicted_score"].sum() * BENCH_WEIGHT
        + (captain_multiplier - 1) * captain_predicted
    )
    captain_objective = df_round.loc[captain_idx, "objective_score"]
    weighted_objective_total = (
        df_round.loc[starters_idx + sixth_idx, "objective_score"].sum() * SCORING_WEIGHT
        + df_round.loc[bench_idx, "objective_score"].sum() * BENCH_WEIGHT
        + (captain_multiplier - 1) * captain_objective
    )
    total_quotation = df_round.loc[roster_idx, "price"].sum()
    print(f"\nTotal quotation used: {total_quotation:.1f} / {BUDGET}")
    print(f"Weighted total predicted score (raw upside): {weighted_predicted_total:.2f}")
    print(f"Weighted total objective score (what the solver actually optimized): {weighted_objective_total:.2f}")
