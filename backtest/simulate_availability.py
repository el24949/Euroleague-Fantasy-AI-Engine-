"""Upper bound: how much would PERFECT availability knowledge help the
engine, using data we already have?

backtest/simulate_many.py measures the engine against two baselines across
rounds 12-34. This file runs that exact same per-round pipeline twice --
once with the normal candidate pool, once with a candidate pool that
additionally excludes any player whose ACTUAL minutes that round were 0 (a
DNP) -- and compares the engine's average actual points between the two.
No DB writes -- read-only, report only.

THIS IS DELIBERATELY NOT LEAKAGE-SAFE FOR AVAILABILITY, ONLY FOR
PERFORMANCE. build_predictions() (reused unchanged) still trains the model
and computes every rolling feature using only Season 2025 games strictly
before the target round -- the leakage barrier from backtest/simulate_round.py
is untouched. But knowing in advance that a specific player will register 0
minutes IS the round's real outcome for that player -- no real fantasy
manager has this before tip-off (injuries, rest, coach's decision, etc.
aren't knowable with certainty ahead of time). This file's whole point is
to intentionally break that one assumption, in isolation, to measure a
ceiling: "if the engine had perfect DNP information -- but NOT perfect
performance predictions -- how much better would it do?" It is not a
proposal for how the live optimizer should work; data/unavailable.csv
(hand-edited, best-effort out/doubtful info) is the realistic version of
this idea.

Reuses simulate_many.simulate_round() (itself built on build_predictions()
and solve_lineup()) via its filter_dnp parameter -- no MILP or
leakage-barrier logic is duplicated here. filter_dnp=False is the exact
same computation simulate_many.py already runs, so the "without" column
below should reproduce those numbers exactly.
"""

import numpy as np
import pandas as pd

from backtest.simulate_many import MAX_ROUND, MIN_ROUND, get_round_gamecode_map, simulate_round
from backtest.simulate_round import SEASON, load_season_boxscores
from data_collection.load_prices import load_prices

if __name__ == "__main__":
    round_map = get_round_gamecode_map(SEASON)
    boxscores = load_season_boxscores()
    prices_df, _ = load_prices()

    rows = []
    for round_number in range(MIN_ROUND, MAX_ROUND + 1):
        gamecodes = round_map.get(round_number)
        if not gamecodes:
            print(f"Round {round_number}: no games found in the schedule, skipping")
            continue

        # Same seed for both runs of this round (fresh generator each time)
        # so the naive/random objective columns solve_lineup uses
        # internally are identical between the two -- the DNP filter is the
        # only thing that can differ between "without" and "with".
        without = simulate_round(
            round_number, gamecodes, boxscores, prices_df,
            np.random.default_rng(42 + round_number), filter_dnp=False,
        )
        with_filter = simulate_round(
            round_number, gamecodes, boxscores, prices_df,
            np.random.default_rng(42 + round_number), filter_dnp=True,
        )

        print(
            f"Round {round_number}: engine without-filter "
            f"{without['engine_actual']:.1f}, with-filter "
            f"{with_filter['engine_actual']:.1f} "
            f"({with_filter['excluded_dnp']} DNP candidate(s) excluded)"
        )
        rows.append(
            {
                "round": round_number,
                "engine_without_filter": without["engine_actual"],
                "engine_with_filter": with_filter["engine_actual"],
                "excluded_dnp": with_filter["excluded_dnp"],
                "engine_roster_with_filter": with_filter["engine_roster"],
            }
        )

    results = pd.DataFrame(rows)
    results["improvement"] = results["engine_with_filter"] - results["engine_without_filter"]

    print("\nPer-round results:")
    display_cols = [
        "round",
        "engine_without_filter",
        "engine_with_filter",
        "improvement",
        "excluded_dnp",
    ]
    print(results[display_cols].to_string(index=False))

    print(f"\nEngine average actual points across {len(results)} rounds ({MIN_ROUND}-{MAX_ROUND}):")
    print(f"  WITHOUT perfect-availability filter: {results['engine_without_filter'].mean():.2f}")
    print(f"  WITH perfect-availability filter:    {results['engine_with_filter'].mean():.2f}")
    print(f"  Average improvement:                 {results['improvement'].mean():+.2f} points/round")
    improved = int((results["improvement"] > 0).sum())
    print(f"  Rounds improved: {improved} / {len(results)}")
