"""Extends optimizer/pick_team.py with a real ILP solve (PuLP) that adds a
budget constraint on top of the roster-shape constraint. Once price enters
the picture, "take the top N per position" is no longer optimal -- price
and predicted score trade off against each other across positions, so we
need an actual solver rather than the greedy per-group nlargest used in
pick_team.py.

Reuses ROUND_GAMECODES, POSITION_REQUIREMENTS, and attach_positions from
optimizer/pick_team.py, and train_and_predict from prediction/model_v3.py
-- nothing here is recomputed. Positions are now real (see pick_team.py's
docstring), not a placeholder.

Also enforces the max-6-players-per-club constraint (data requirements
doc Section 2). This uses the real Team column from the box scores --
unlike prices, club is not a placeholder.

PLACEHOLDER: real fantasy prices (data requirement row 5) don't exist yet.
Prices below are a monotonic function of predicted_score (better predicted
players cost more) plus small deterministic noise from Player_ID, scaled
so a full team of the best predicted scorers costs well over budget --
i.e. the budget constraint is genuinely binding, not automatically
satisfied. This has no relationship to real EuroLeague Fantasy prices and
must be replaced once real price data lands.
"""

import hashlib

import pulp

from optimizer.pick_team import POSITION_REQUIREMENTS, ROUND_GAMECODES, attach_positions
from prediction.model_v3 import train_and_predict

BUDGET = 100
MAX_PER_CLUB = 6


def dummy_price(player_id, predicted_score):
    base_price = 4.0 + 0.6 * predicted_score
    digest = hashlib.md5((player_id + "_price").encode()).hexdigest()
    noise = (int(digest, 16) % 21 - 10) / 10  # deterministic, in [-1.0, 1.0]
    price = max(1.0, base_price + noise)
    return round(price, 1)


if __name__ == "__main__":
    model, feature_cols, train, test = train_and_predict()

    df_round = test[test["Gamecode"].isin(ROUND_GAMECODES)].copy()
    df_round = attach_positions(df_round)
    df_round["price"] = df_round.apply(
        lambda row: dummy_price(row["Player_ID"], row["predicted_score"]), axis=1
    )

    prob = pulp.LpProblem("euroleague_fantasy_team", pulp.LpMaximize)

    player_vars = {
        idx: pulp.LpVariable(f"select_{idx}", cat="Binary") for idx in df_round.index
    }

    prob += pulp.lpSum(
        player_vars[idx] * df_round.loc[idx, "predicted_score"] for idx in df_round.index
    )

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

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    selected_idx = [idx for idx in df_round.index if player_vars[idx].value() == 1]
    team = df_round.loc[selected_idx]

    print(f"Solver status: {status}")
    print(
        team[["Player", "position", "Team", "price", "predicted_score"]].to_string(index=False)
    )
    print(f"\nPlayers per club:\n{team['Team'].value_counts().to_string()}")
    print(f"\nTotal price used: {team['price'].sum():.1f} / {BUDGET}")
    print(f"Total predicted score: {team['predicted_score'].sum():.2f}")
