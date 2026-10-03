"""Trade suggestions: given my CURRENT team (optimizer/my_team.py's
load_my_team()), find the best set of up to N one-for-one swaps that
maximizes the team's risk-aware projected score.

This is genuinely NEW logic -- optimizer/pick_lineup_ilp.py's solve_lineup()
picks a legal team FROM SCRATCH out of a candidate pool; it has no concept of
"my current roster" or "how many changes am I allowed", so it can't answer
"what should I trade". Everything feeding the objective, though, is reused as-
is: predicted_score/consistency_5 come from run_prediction_step (pipeline/
update_round.py), the risk-aware objective is RISK_LAMBDA * consistency_5
(optimizer/pick_lineup_ilp.py), and the roster-shape/budget/club constraints
are POSITION_REQUIREMENTS (optimizer/pick_team.py) and BUDGET/MAX_PER_CLUB
(optimizer/pick_team_ilp.py).

TRADE DEFINITION: 1 trade = remove 1 current player, add 1 new player
(one-for-one). A coach change also counts as 1 trade. max_trades is an int
cap, or None for an unlimited-trade round.

BLIND PLAYERS ARE PINNED, NOT SWAPPABLE: a current player with no prediction
this round (not enough prior history) has no objective_score. Letting the
solver treat that as "0, so swap them for literally anyone" would be a
modeling artifact, not a real judgement -- it would force blind players out
on every run regardless of their real quality. So they're pinned as a FIXED
part of the resulting team (never a swap-out candidate) and never offered as
a swap-in either (same reason: nothing to rate them on). This is what makes
"never force a blind player out" true by construction rather than by a
heuristic.

COACH IS NEVER SEARCHED HERE: the only existing coach-value signal
(optimizer/coach_ranking.rank_coaches's recent_avg_margin) is real-game point
margin, not commensurable with the player objective's fantasy-point units --
folding them into one objective would mean inventing a new metric, not reusing
one. coach_change is always reported as unchanged.

ILP FORMULATION (PuLP/CBC, same pattern as optimizer/pick_team_ilp.py):
  - keep_<id> (binary) for each current player WITH a prediction: stays in
    the team if 1.
  - add_<id> (binary) for each candidate player (price pool, has a
    prediction, not already on the roster): joins the team if 1.
  - roster size: n_blind + sum(keep) + sum(add) == 10 (this is what forces
    sum(add) == number of players dropped, i.e. true 1-for-1 swaps).
  - position counts, club limit (<=6), and total cost (players + coach
    price) <= budget are each: blind players' fixed contribution + the
    decision-variable sum, matching the required/limit value.
  - trade cap: sum(add) <= max_trades (skipped entirely if max_trades is
    None).
  - objective: maximize sum(keep * objective_score) + sum(add *
    objective_score). Pinned blind players contribute a constant (0,
    since unrated) to both the current and projected score, so the
    reported DELTA between them is still exact.
"""

import pulp

from optimizer.my_team import load_my_team, load_price_pool
from optimizer.pick_lineup_ilp import RISK_LAMBDA
from optimizer.pick_team import POSITION_REQUIREMENTS
from optimizer.pick_team_ilp import BUDGET, MAX_PER_CLUB
from pipeline.update_round import get_round_gamecodes, run_prediction_step


def _score_round_features(round_features):
    """round_features (run_prediction_step's return) reduced to
    Player_ID/objective_score, deduplicated and whitespace-stripped on
    Player_ID -- this project's one recurring join bug (see CLAUDE.md's
    "the whitespace bug")."""
    scored = round_features[["Player_ID", "predicted_score", "consistency_5"]].copy()
    scored["Player_ID"] = scored["Player_ID"].str.strip()
    scored["objective_score"] = (
        scored["predicted_score"] - RISK_LAMBDA * scored["consistency_5"]
    )
    return scored.drop_duplicates(subset="Player_ID")


def _prepare_swap_pool(price_pool, exclude_ids):
    """load_price_pool()'s full pool, reduced to real (non-placeholder)
    players not already on the current roster, with a display name in the
    same 'SURNAME, NAME' convention load_my_team()/my_team.csv use."""
    pool = price_pool[price_pool["Player_ID"].notna()].copy()
    pool["Player_ID"] = pool["Player_ID"].str.strip()
    pool = pool.rename(columns={"Quotation": "price"})
    pool["display_name"] = pool["Surname"].str.strip() + ", " + pool["Name"].str.strip()
    return pool[~pool["Player_ID"].isin(exclude_ids)]


def suggest_trades(current_team, max_trades, season, round_no, budget=BUDGET):
    """Best up to-max_trades one-for-one player swaps for current_team
    (load_my_team()'s return: 'players' DataFrame + 'coach' dict + 'state').
    Coach is never changed -- see module docstring. Returns a plain dict,
    see the module docstring's ILP section for how it's built."""
    players = current_team["players"].copy()
    players["Player_ID"] = players["Player_ID"].str.strip()
    coach = current_team["coach"]

    round_features = run_prediction_step(get_round_gamecodes(season, round_no))
    rated = _score_round_features(round_features)

    rated_current = players.merge(rated, on="Player_ID", how="inner")
    blind_current = players[~players["Player_ID"].isin(rated_current["Player_ID"])]

    price_pool = load_price_pool()
    swap_pool = _prepare_swap_pool(price_pool, players["Player_ID"])
    swap_candidates = swap_pool.merge(rated, on="Player_ID", how="inner")
    swap_candidates = swap_candidates.drop_duplicates(subset="Player_ID")

    rated_current_by_id = rated_current.set_index("Player_ID")
    swap_candidates_by_id = swap_candidates.set_index("Player_ID")

    keep_vars = {
        pid: pulp.LpVariable(f"keep_{pid}", cat="Binary") for pid in rated_current_by_id.index
    }
    add_vars = {
        pid: pulp.LpVariable(f"add_{pid}", cat="Binary") for pid in swap_candidates_by_id.index
    }

    objective_expr = pulp.lpSum(
        keep_vars[pid] * rated_current_by_id.loc[pid, "objective_score"] for pid in keep_vars
    ) + pulp.lpSum(
        add_vars[pid] * swap_candidates_by_id.loc[pid, "objective_score"] for pid in add_vars
    )
    blind_cost = float(blind_current["price"].sum())

    def _add_shared_constraints(prob):
        """Roster-size/position/club/budget/trade-cap constraints, shared
        identically by both solve phases below (phase 1 maximizes score;
        phase 2 re-solves for the fewest-trades tie-break) -- written once
        so the two phases can never drift apart."""
        n_blind = len(blind_current)
        prob += pulp.lpSum(keep_vars.values()) + pulp.lpSum(add_vars.values()) == 10 - n_blind

        for position, required in POSITION_REQUIREMENTS.items():
            blind_count = int((blind_current["position"] == position).sum())
            prob += (
                blind_count
                + pulp.lpSum(
                    keep_vars[pid] for pid in keep_vars
                    if rated_current_by_id.loc[pid, "position"] == position
                )
                + pulp.lpSum(
                    add_vars[pid] for pid in add_vars
                    if swap_candidates_by_id.loc[pid, "position"] == position
                )
                == required
            )

        clubs = set(blind_current["Team"]) | set(rated_current["Team"]) | set(swap_candidates["Team"])
        for club in clubs:
            blind_count = int((blind_current["Team"] == club).sum())
            prob += (
                blind_count
                + pulp.lpSum(
                    keep_vars[pid] for pid in keep_vars
                    if rated_current_by_id.loc[pid, "Team"] == club
                )
                + pulp.lpSum(
                    add_vars[pid] for pid in add_vars
                    if swap_candidates_by_id.loc[pid, "Team"] == club
                )
                <= MAX_PER_CLUB
            )

        prob += (
            blind_cost
            + pulp.lpSum(keep_vars[pid] * rated_current_by_id.loc[pid, "price"] for pid in keep_vars)
            + pulp.lpSum(add_vars[pid] * swap_candidates_by_id.loc[pid, "price"] for pid in add_vars)
            + float(coach["price"])
            <= budget
        )

        # max_trades is an UPPER BOUND, never a required count -- keeping a
        # current player (0 trades) is always legal. "<=", never "==".
        if max_trades is not None:
            prob += pulp.lpSum(add_vars.values()) <= max_trades

    # --- Phase 1: find the best achievable projected score. ---
    prob = pulp.LpProblem("suggest_trades", pulp.LpMaximize)
    prob += objective_expr
    _add_shared_constraints(prob)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    current_score = round(float(rated_current["objective_score"].sum()), 2)
    unchanged_result = {
        "swaps": [],
        "coach_change": {"changed": False, "from": coach["name"], "to": coach["name"]},
        "current_score": current_score,
        "projected_score": current_score,
        "trades_used": 0,
        "max_trades": max_trades,
        "resulting_total_cost": round(float(players["price"].sum() + coach["price"]), 1),
        "solver_status": status,
    }
    if status != "Optimal":
        return unchanged_result

    # --- Phase 2: among every solution that reaches phase 1's best score
    # (within floating-point tolerance), pick the one using the FEWEST
    # trades -- never spend a trade that doesn't improve the team. This is
    # the fewer-trades tie-break: lock the score at its just-found optimum,
    # then re-solve minimizing sum(add_vars) subject to that. Same shared
    # constraints (incl. the "<=" trade cap), so 0 trades stays legal here
    # too if it already matched the best score.
    TOLERANCE = 1e-6
    best_score = pulp.value(prob.objective)

    tiebreak_prob = pulp.LpProblem("suggest_trades_fewest_trades", pulp.LpMinimize)
    tiebreak_prob += pulp.lpSum(add_vars.values())
    _add_shared_constraints(tiebreak_prob)
    tiebreak_prob += objective_expr >= best_score - TOLERANCE
    tiebreak_prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[tiebreak_prob.status]

    dropped = [pid for pid in keep_vars if keep_vars[pid].value() == 0]
    added = [pid for pid in add_vars if add_vars[pid].value() == 1]
    if not dropped and not added:
        return unchanged_result

    # Pairing order is cosmetic only (the ILP optimized jointly, not pair by
    # pair) -- worst-out-first / best-in-first just reads more sensibly, and
    # each pair's gain still sums to the exact total gain either way.
    dropped_sorted = sorted(dropped, key=lambda pid: rated_current_by_id.loc[pid, "objective_score"])
    added_sorted = sorted(
        added, key=lambda pid: swap_candidates_by_id.loc[pid, "objective_score"], reverse=True
    )

    swaps = []
    for out_pid, in_pid in zip(dropped_sorted, added_sorted):
        out_row = rated_current_by_id.loc[out_pid]
        in_row = swap_candidates_by_id.loc[in_pid]
        swaps.append(
            {
                "out_player": out_row["player_name"],
                "in_player": in_row["display_name"],
                "out_price": float(out_row["price"]),
                "in_price": float(in_row["price"]),
                # Separate fields -- a pairing is cosmetic (see the sort above),
                # so the out and in player are NOT guaranteed to share a
                # position; collapsing both into one "position" field silently
                # mislabeled the in player whenever they differed.
                "out_position": out_row["position"],
                "in_position": in_row["position"],
                "projected_gain": round(float(in_row["objective_score"] - out_row["objective_score"]), 2),
            }
        )

    projected_score = round(current_score + sum(s["projected_gain"] for s in swaps), 2)

    resulting_cost = blind_cost + float(coach["price"])
    for pid in keep_vars:
        if keep_vars[pid].value() == 1:
            resulting_cost += float(rated_current_by_id.loc[pid, "price"])
    for pid in added:
        resulting_cost += float(swap_candidates_by_id.loc[pid, "price"])

    return {
        "swaps": swaps,
        "coach_change": {"changed": False, "from": coach["name"], "to": coach["name"]},
        "current_score": current_score,
        "projected_score": projected_score,
        "trades_used": len(swaps),
        "max_trades": max_trades,
        "resulting_total_cost": round(resulting_cost, 1),
        "solver_status": status,
    }


if __name__ == "__main__":
    my_team = load_my_team()

    # run_prediction_step/build_predictions and rank_coaches are internally
    # hardwired to Season 2025 box scores (not parameterized) -- a pre-
    # existing limitation, out of scope for this slice. Season 2026 has no
    # games played yet (CLAUDE.md's "2026 LAUNCH REALITY"), so a 2026 round
    # would just exercise the all-blind no-op path. Round 32 matches
    # pipeline/update_round.py's own TARGET_ROUND, for a realistic test.
    SEASON = 2025
    ROUND_NO = 32
    MAX_TRADES = 4

    result = suggest_trades(my_team, MAX_TRADES, SEASON, ROUND_NO)

    print(f"Season {SEASON}, Round {ROUND_NO} -- max_trades={MAX_TRADES}")
    print(f"Solver status: {result['solver_status']}")
    print(f"Current score: {result['current_score']:.2f}")
    print(f"Projected score after trades: {result['projected_score']:.2f}")
    print(f"Trades used: {result['trades_used']} / {result['max_trades']}")
    print(f"Resulting total cost: {result['resulting_total_cost']:.1f} / {BUDGET}")

    if result["swaps"]:
        print("\nSuggested swaps:")
        for swap in result["swaps"]:
            print(
                f"  OUT {swap['out_player']} ({swap['out_position']}, {swap['out_price']:.1f}) "
                f"-> IN {swap['in_player']} ({swap['in_position']}, {swap['in_price']:.1f}) "
                f"[gain {swap['projected_gain']:+.2f}]"
            )
    else:
        print("\nNo swaps suggested.")

    coach_change = result["coach_change"]
    print(f"\nCoach unchanged: {coach_change['from']}")
