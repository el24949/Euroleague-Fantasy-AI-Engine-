"""Extends optimizer/pick_team_ilp.py with the actual scoring lineup, per
data requirements doc Sections 1-2: a 10-man roster (4G/4F/2C, budget<=100,
max 6/club, unchanged from pick_team_ilp.py), from which a starting five +
6th man score at 100% and the remaining 4 bench players score at 50%. The
starting five must match one legal formation (G-F-C counts); the 6th man
has no position restriction.

This is one combined MILP, not two separate stages -- the roster and the
lineup are optimized jointly against the weighted objective below, so the
solver can trade off "who's on the roster" against "who starts" together.

Reuses ROUND_GAMECODES, POSITION_REQUIREMENTS, placeholder_position
(optimizer/pick_team.py), BUDGET, MAX_PER_CLUB, dummy_price
(optimizer/pick_team_ilp.py), and train_and_predict (prediction/model_v3.py)
-- nothing here is recomputed. Positions and prices remain placeholders,
club is real data, same as pick_team_ilp.py.
"""

import pulp

from optimizer.pick_team import POSITION_REQUIREMENTS, ROUND_GAMECODES, placeholder_position
from optimizer.pick_team_ilp import BUDGET, MAX_PER_CLUB, dummy_price
from prediction.model_v3 import train_and_predict

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

    df_round = test[test["Gamecode"].isin(ROUND_GAMECODES)].copy()
    df_round["position"] = df_round["Player_ID"].apply(placeholder_position)
    df_round["price"] = df_round.apply(
        lambda row: dummy_price(row["Player_ID"], row["predicted_score"]), axis=1
    )

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

    # --- objective: 100% scoring group (starters + 6th man) + 50% bench ---
    scoring_group_score = pulp.lpSum(
        df_round.loc[idx, "predicted_score"] * (starter_vars[idx] + sixth_man_vars[idx])
        for idx in df_round.index
    )
    bench_score = pulp.lpSum(
        df_round.loc[idx, "predicted_score"]
        * (player_vars[idx] - starter_vars[idx] - sixth_man_vars[idx])
        for idx in df_round.index
    )
    prob += SCORING_WEIGHT * scoring_group_score + BENCH_WEIGHT * bench_score

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

    columns = ["Player", "position", "predicted_score"]

    print(f"Solver status: {status}")
    print(f"Formation: {active_formation}")

    print("\nStarting five:")
    print(df_round.loc[starters_idx, columns].to_string(index=False))

    print("\n6th man:")
    print(df_round.loc[sixth_idx, columns].to_string(index=False))

    print("\nBench (50%):")
    print(df_round.loc[bench_idx, columns].to_string(index=False))

    weighted_total = (
        df_round.loc[starters_idx + sixth_idx, "predicted_score"].sum() * SCORING_WEIGHT
        + df_round.loc[bench_idx, "predicted_score"].sum() * BENCH_WEIGHT
    )
    print(f"\nWeighted total predicted score: {weighted_total:.2f}")
