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

RISK-AWARE OBJECTIVE (optional): solve_engine()'s lambda_risk parameter
lets the engine optimize objective_score = predicted_score - lambda_risk *
consistency_5 instead of raw predicted_score. consistency_5 (a player's
rolling std of fantasy score) is already a leakage-free feature computed
by build_predictions() -- see features/rolling_form.py -- never
recomputed here, just subtracted in with a weight. lambda_risk=0.0 (the
default, used everywhere except the sweep at the bottom of __main__) skips
building objective_score entirely and solves against predicted_score
directly, so it's bit-identical to the engine before this option existed.
simulate_round_multi_lambda() builds one round's candidate pool ONCE (model
training/features don't depend on lambda_risk at all) and only re-solves
the engine per LAMBDA, instead of rerunning the whole leakage-safe pipeline
once per LAMBDA for no reason.

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
LAMBDA_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]


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


def prepare_round(round_number, gamecodes, boxscores, prices_df, rng, filter_dnp=False):
    """Build one round's leakage-safe candidate pool: predictions, real
    prices/positions, the naive baseline's prior-average, a random draw,
    and the round's real actual score -- everything needed to solve ANY
    engine objective, plus the naive/random baselines. Returns (df_round,
    captain_multiplier, excluded_no_price, excluded_dnp).

    Split out from simulate_round() so callers that need MULTIPLE engine
    objectives for the same round (simulate_round_multi_lambda()) can call
    this ONCE and only re-solve the engine's MILP per objective --
    build_predictions() (the leakage barrier: train/features from only
    strictly-prior Season 2025 games) doesn't depend on the engine's
    objective at all, so recomputing it per objective would be pure waste.

    filter_dnp=True additionally drops any candidate whose ACTUAL minutes
    this round were 0 (a DNP) before price/naive/random are even computed
    -- see backtest/simulate_availability.py's docstring for why this is
    the one deliberate exception to the leakage barrier (an upper-bound
    experiment on availability info, not a real prediction). Default False
    leaves this identical to before filter_dnp existed.
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

    return df_round, captain_multiplier, excluded_no_price, excluded_dnp


def solve_engine(df_round, captain_multiplier, lambda_risk=0.0):
    """Solve the engine's lineup, optionally risk-adjusted: objective_score
    = predicted_score - lambda_risk * consistency_5. consistency_5 (a
    player's rolling std of fantasy score) is already a leakage-free
    feature on df_round -- see features/rolling_form.py -- never
    recomputed here, just weighted in.

    lambda_risk=0.0 (falsy) skips building objective_score entirely and
    solves against predicted_score directly -- not "predicted_score minus
    zero" -- so this is bit-identical to the engine before lambda_risk
    existed, which is the required control case.
    """
    if lambda_risk:
        df_round = df_round.copy()
        df_round["objective_score"] = (
            df_round["predicted_score"] - lambda_risk * df_round["consistency_5"]
        )
        score_col = "objective_score"
    else:
        score_col = "predicted_score"
    return solve_lineup(df_round, score_col, captain_multiplier)


def simulate_round(round_number, gamecodes, boxscores, prices_df, rng, filter_dnp=False, lambda_risk=0.0):
    """Run the full leakage-safe pipeline for one round: build predictions
    from only-prior data, join real prices/positions, solve engine/naive/
    random lineups under identical constraints, score all three on the
    round's real outcome. Returns one results row (dict).

    See prepare_round() for filter_dnp and solve_engine() for lambda_risk.
    Both default to their no-op values, so simulate_round(...) with neither
    argument behaves exactly as it did before either option existed.
    """
    df_round, captain_multiplier, excluded_no_price, excluded_dnp = prepare_round(
        round_number, gamecodes, boxscores, prices_df, rng, filter_dnp=filter_dnp
    )

    engine_result = solve_engine(df_round, captain_multiplier, lambda_risk)
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


def simulate_round_multi_lambda(round_number, gamecodes, boxscores, prices_df, rng, lambda_values):
    """Prepare one round's candidate pool ONCE, then re-solve ONLY the
    engine for each LAMBDA in lambda_values (the naive baseline doesn't
    depend on the engine's objective, so it's solved once and reused).
    Returns {lambda_risk: {"engine_actual": ..., "engine_beats_naive": ...}}.
    """
    df_round, captain_multiplier, _, _ = prepare_round(
        round_number, gamecodes, boxscores, prices_df, rng
    )

    naive_result = solve_lineup(df_round, "prior_avg_score", captain_multiplier)
    assert_roster_complete("Naive", df_round, naive_result)
    naive_actual = score_team(df_round, naive_result, captain_multiplier, "actual_score")

    per_lambda = {}
    for lambda_risk in lambda_values:
        engine_result = solve_engine(df_round, captain_multiplier, lambda_risk)
        assert_roster_complete(f"Engine (lambda={lambda_risk})", df_round, engine_result)
        engine_actual = score_team(df_round, engine_result, captain_multiplier, "actual_score")
        per_lambda[lambda_risk] = {
            "engine_actual": engine_actual,
            "engine_beats_naive": engine_actual > naive_actual,
        }
    return per_lambda


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

    # --- risk-aware objective sweep: objective_score = predicted_score -
    # --- LAMBDA * consistency_5. LAMBDA=0.0 is the control -- it must
    # --- reproduce the "Engine" row above exactly, since solve_engine()
    # --- solves against predicted_score directly (not predicted_score
    # --- minus zero) whenever lambda_risk is falsy ---
    print(f"\n{'-' * 60}")
    print(f"Risk-aware objective sweep: objective_score = predicted_score - LAMBDA * consistency_5")
    print(f"LAMBDA values: {LAMBDA_VALUES}\n")

    sweep_rng = np.random.default_rng(42)
    per_lambda_rows = {lam: [] for lam in LAMBDA_VALUES}
    for round_number in range(MIN_ROUND, MAX_ROUND + 1):
        gamecodes = round_map.get(round_number)
        if not gamecodes:
            continue
        per_lambda = simulate_round_multi_lambda(
            round_number, gamecodes, boxscores, prices_df, sweep_rng, LAMBDA_VALUES
        )
        for lam, row in per_lambda.items():
            per_lambda_rows[lam].append(row)

    sweep_summary = []
    for lam in LAMBDA_VALUES:
        rows_for_lambda = per_lambda_rows[lam]
        n = len(rows_for_lambda)
        avg_actual = sum(r["engine_actual"] for r in rows_for_lambda) / n
        beats_naive = sum(r["engine_beats_naive"] for r in rows_for_lambda)
        sweep_summary.append(
            {
                "LAMBDA": lam,
                "avg_actual_points": round(avg_actual, 2),
                "rounds_beating_naive": f"{beats_naive}/{n}",
            }
        )

    sweep_df = pd.DataFrame(sweep_summary)
    print(sweep_df.to_string(index=False))

    control_match = sweep_df.loc[sweep_df["LAMBDA"] == 0.0, "avg_actual_points"].iloc[0]
    assert control_match == round(results["engine_actual"].mean(), 2), (
        f"LAMBDA=0.0 control ({control_match}) does not match the plain engine "
        f"average ({results['engine_actual'].mean():.2f}) -- risk-aware sweep "
        f"must be broken"
    )
    print(f"\nControl check: LAMBDA=0.0 matches the plain engine average above ({control_match}). OK.")
