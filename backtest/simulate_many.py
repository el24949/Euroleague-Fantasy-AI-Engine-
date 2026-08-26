"""Multi-round leakage-safe backtest for Season 2025.

backtest/simulate_round.py answers "how did the engine do in one round?".
This file asks the same question across every round from MIN_ROUND to
MAX_ROUND, retraining on only-prior-rounds each time (the model for round
20 has never seen round 20 or later; the model for round 21 has never seen
round 21 or later, etc.), so a single lucky/unlucky round can't dominate
the read on whether the engine actually beats simple baselines. No DB
writes -- read-only, report only.

Reuses everything from backtest/simulate_round.py -- build_predictions()
(the leakage barrier: blind the target round's stats, train only on
strictly-prior Season 2025 games, predict with zero visibility into the
real outcome), score_team(), assert_roster_complete(), load_season_boxscores(),
get_round_name(), and the SEASON/FEATURE_COLS constants -- plus solve_lineup()
(optimizer/pick_lineup_ilp.py) for the roster+lineup MILP. The only new
code here is the per-round loop, the round-number -> Gamecode lookup, and
the results table. Nothing about the leakage-safe pipeline itself is
reimplemented.

ROUND RANGE: MIN_ROUND=12, MAX_ROUND=34. Round 12 is the first round with
a substantial prior history (110 games from rounds 1-11) -- earlier rounds
train on very little data, and many candidates are still mid-cold-start
(several rolling features not yet non-NaN). MAX_ROUND=34 is not a hard
technical boundary -- rounds 35-38 have plenty of history behind them too
-- it's simply the requested range; raise MAX_ROUND to 38 to cover the
rest of the season.

Round -> Gamecode mapping comes from euroleague_api's Schedule (the same
lookup simulate_round.get_round_name uses), not assumed from a fixed
offset -- Season 2025's regular-season rounds happen to each be exactly 10
consecutive Gamecodes, but this looks that up rather than hardcoding it.
"""

import numpy as np
import pandas as pd
from euroleague_api.schedule import Schedule

from backtest.simulate_round import (
    SEASON,
    assert_roster_complete,
    build_predictions,
    load_season_boxscores,
    score_team,
)
from data_collection.load_prices import load_prices
from features.minutes import minutes_to_float
from optimizer.pick_lineup_ilp import solve_lineup
from optimizer.pick_team import attach_positions, get_round_season
from scoring.base_score import base_fantasy_score
from scoring.season_rules import get_rules

MIN_ROUND = 12
MAX_ROUND = 34


def get_round_gamecode_map(season):
    """{round_number: [gamecodes]} for every real-season round, via
    euroleague_api's Schedule -- looked up rather than assumed from a
    fixed offset."""
    schedule_df = Schedule().get_schedule(season)
    schedule_df = schedule_df[schedule_df["round"] == "RS"].copy()
    schedule_df["game"] = schedule_df["game"].astype(int)
    return schedule_df.groupby("gameday")["game"].apply(lambda s: sorted(s.tolist())).to_dict()


def format_roster_names(df_round, result):
    """Abbreviated 'F.Surname' names for a solved lineup (boxscores.Player
    is 'SURNAME, FIRSTNAME'), captain marked with (C)."""
    names = []
    for idx in result["roster_idx"]:
        surname, firstname = [part.strip() for part in df_round.loc[idx, "Player"].split(",", 1)]
        abbrev = f"{firstname[0]}.{surname.title()}"
        if idx == result["captain_idx"]:
            abbrev += "(C)"
        names.append(abbrev)
    return ", ".join(names)


def simulate_round(round_number, gamecodes, boxscores, prices_df, rng, filter_dnp=False):
    """Run the full leakage-safe pipeline for one round: build predictions
    from only-prior data, join real prices/positions, solve engine/naive/
    random lineups under identical constraints, score all three on the
    round's real outcome. Returns one results row (dict).

    filter_dnp=True additionally drops any candidate whose ACTUAL minutes
    this round were 0 (a DNP) before any team is picked -- see
    backtest/simulate_availability.py's docstring for why this is the one
    deliberate exception to the leakage barrier (an upper-bound experiment
    on availability info, not a real prediction). Default False leaves this
    function's behavior identical to before filter_dnp existed.
    """
    model, history_computed, round_features, round_df = build_predictions(
        gamecodes, boxscores=boxscores
    )

    df_round = attach_positions(round_features)

    # --- OPTIONAL perfect-availability filter: exclude candidates who
    # --- actually registered 0 minutes this round, before price/naive/
    # --- random are even computed, same "drop from pool entirely" treatment
    # --- as the no-price exclusion below ---
    excluded_dnp = 0
    if filter_dnp:
        minutes_lookup = dict(
            zip(round_df["Player_ID"].str.strip(), round_df["Minutes"].apply(minutes_to_float))
        )
        dnp_mask = df_round["Player_ID"].str.strip().map(minutes_lookup) == 0
        excluded_dnp = int(dnp_mask.sum())
        df_round = df_round[~dnp_mask].copy()

    # --- real quotations; candidates with no matching price can't be
    # --- selected by any of the three teams ---
    price_lookup = dict(zip(prices_df["Player_ID"], prices_df["Quotation"]))
    df_round["price"] = df_round["Player_ID"].str.strip().map(price_lookup)
    excluded_no_price = int(df_round["price"].isna().sum())
    df_round = df_round[df_round["price"].notna()].copy()

    # --- naive baseline: each candidate's own Season-2025 average
    # --- base_score across all real games strictly before this round ---
    prior_avg_lookup = (
        history_computed.assign(pid=history_computed["Player_ID"].str.strip())
        .groupby("pid")["base_score"]
        .mean()
    )
    df_round["prior_avg_score"] = df_round["Player_ID"].str.strip().map(prior_avg_lookup)
    missing_prior_avg = df_round["prior_avg_score"].isna()
    if missing_prior_avg.any():
        raise ValueError(
            f"Round {round_number}: candidate(s) with a predicted_score but no "
            f"prior-average score, which should be impossible: "
            f"{df_round.loc[missing_prior_avg, 'Player'].tolist()}"
        )

    # --- random baseline: draws from the single rng threaded across all
    # --- rounds, so the whole multi-round run is reproducible end to end
    # --- (not just each round in isolation) ---
    df_round["random_score"] = rng.random(len(df_round))

    # --- actual points, joined in only AFTER every team below is picked ---
    round_df = round_df.copy()
    round_df["actual_score"] = round_df.apply(base_fantasy_score, axis=1)
    round_df["Player_ID"] = round_df["Player_ID"].str.strip()
    actual_lookup = dict(zip(round_df["Player_ID"], round_df["actual_score"]))
    df_round["actual_score"] = df_round["Player_ID"].str.strip().map(actual_lookup)

    season = get_round_season(df_round)
    captain_multiplier = get_rules(season)["captain_multiplier"]

    engine_result = solve_lineup(df_round, "predicted_score", captain_multiplier)
    naive_result = solve_lineup(df_round, "prior_avg_score", captain_multiplier)
    random_result = solve_lineup(df_round, "random_score", captain_multiplier)

    for label, result in [
        ("Engine", engine_result),
        ("Naive", naive_result),
        ("Random", random_result),
    ]:
        assert_roster_complete(label, df_round, result)

    return {
        "round": round_number,
        "gamecodes": f"{min(gamecodes)}-{max(gamecodes)}",
        "excluded_no_price": excluded_no_price,
        "excluded_dnp": excluded_dnp,
        "engine_actual": score_team(df_round, engine_result, captain_multiplier, "actual_score"),
        "naive_actual": score_team(df_round, naive_result, captain_multiplier, "actual_score"),
        "random_actual": score_team(df_round, random_result, captain_multiplier, "actual_score"),
        "engine_roster": format_roster_names(df_round, engine_result),
        "naive_roster": format_roster_names(df_round, naive_result),
        "random_roster": format_roster_names(df_round, random_result),
    }


if __name__ == "__main__":
    round_map = get_round_gamecode_map(SEASON)
    boxscores = load_season_boxscores()
    prices_df, _ = load_prices()
    rng = np.random.default_rng(42)

    rows = []
    for round_number in range(MIN_ROUND, MAX_ROUND + 1):
        gamecodes = round_map.get(round_number)
        if not gamecodes:
            print(f"Round {round_number}: no games found in the schedule, skipping")
            continue
        row = simulate_round(round_number, gamecodes, boxscores, prices_df, rng)
        print(
            f"Round {round_number} (Gamecode {row['gamecodes']}): engine "
            f"{row['engine_actual']:.1f}, naive {row['naive_actual']:.1f}, "
            f"random {row['random_actual']:.1f}"
        )
        rows.append(row)

    results = pd.DataFrame(rows)
    results["engine_beats_naive"] = results["engine_actual"] > results["naive_actual"]

    print("\nPer-round results:")
    display_cols = [
        "round",
        "gamecodes",
        "engine_actual",
        "naive_actual",
        "random_actual",
        "engine_beats_naive",
        "engine_roster",
        "naive_roster",
        "random_roster",
    ]
    print(results[display_cols].to_string(index=False))

    beats = int(results["engine_beats_naive"].sum())
    ties = int((results["engine_actual"] == results["naive_actual"]).sum())
    print(f"\nAverage actual points across {len(results)} rounds ({MIN_ROUND}-{MAX_ROUND}):")
    print(f"  Engine: {results['engine_actual'].mean():.2f}")
    print(f"  Naive:  {results['naive_actual'].mean():.2f}")
    print(f"  Random: {results['random_actual'].mean():.2f}")
    print(f"\nEngine beat naive in {beats} / {len(results)} rounds ({ties} tie(s))")
