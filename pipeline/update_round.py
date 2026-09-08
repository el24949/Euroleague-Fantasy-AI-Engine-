"""Per-round data-refresh orchestrator -- the go-live update routine.

Run this before picking a team each round to bring data/euroleague.db up to
date for SEASON (2025, the current working season), then generate the
actual recommended team. In order:
  1. box-score collector (data_collection.fetch_season.collect_season)
  2. positions collector (data_collection.fetch_positions, run via import)
  3. turn-tag refresh (features.turn_tags.build_turn_lookup)
  4. price loader for the target round (data_collection.load_prices_history)
  5. leakage-safe feature pipeline + prediction (only games strictly
     before the target round)
  6. the live optimizer (positions/prices/turns/coach/RISK_LAMBDA/
     turn-coverage, exactly as optimizer/pick_lineup_ilp.py runs it) and
     the final 11-slot team printout
Steps 1-3 also produce a status summary: latest round/gamecode present in
the DB, total games, and which round is 'next/upcoming'.

WHERE THE MANUAL PRICE STEP BELONGS: unlike box scores/positions/turns,
there's no API for fantasy prices (login-gated) -- a person has to
download that round's price Excel from the EuroLeague Fantasy site by
hand and drop it at data/prices/round_NN/players_stats.xlsx before
data_collection/load_prices_history.py (step 4) can load it (see that
file's own docstring). status_summary() (steps 1-3) prints where that file
needs to go using the next_round it just computed, so the manual step
between "data refreshed" and "team generated" stays visible even though
this file can't do the download itself.

TESTING ON A PAST ROUND (Season 2025 is fully played, so there is no real
"upcoming round" right now): TARGET_ROUND below picks one already-played
round and step 5 treats it exactly the way backtest/simulate_round.py
treats its target round -- every feature and the model are computed using
ONLY Season 2025 games with Gamecode strictly before it (see that file's
own STRICT LEAKAGE RULE). TARGET_ROUND=32 matches
optimizer/pick_team.py's ROUND_GAMECODES_2025 (Gamecode 311-320), the same
round every other file in this codebase (pick_lineup_ilp.py, turn_join.py,
coach_ranking.py) already uses as its own test target -- kept in sync here
via get_round_gamecodes(SEASON, TARGET_ROUND) rather than importing that
range directly, so this file works for any round number, not just 32.
PRICE-DATA CAVEAT: the only real price file this project has
(data/prices/players_stats.xlsx, the same one load_prices() reads
everywhere else) was copied to data/prices/round_32/players_stats.xlsx to
exercise step 4 -- real Player_IDs/positions/quotations, but NOT verified
to be round 32's actual historical prices (there is no way to check that
without a second, independently-dated source). Good enough to prove the
pipeline wiring end-to-end; not a claim about round-accurate pricing.

Reuses collect_season (data_collection/fetch_season.py), fetch_positions.py
(data_collection/fetch_positions.py, imported rather than called -- see
STEP 2 below), build_turn_lookup/load_regular_season_schedule
(features/turn_tags.py), load_prices_for_round/store_prices
(data_collection/load_prices_history.py), build_predictions
(backtest/simulate_round.py -- the same leakage-safe feature+model logic
model_v3.py uses, parameterized on an arbitrary target round instead of
model_v3's fixed CUTOFF_GAMECODE split), and attach_positions/
get_round_season (optimizer/pick_team.py), load_unavailable
(optimizer/availability.py), attach_turns (optimizer/turn_join.py),
rank_coaches/SHORTLIST_SIZE (optimizer/coach_ranking.py), get_rules
(scoring/season_rules.py), solve_lineup/RISK_LAMBDA/TURN_COVERAGE_WEIGHT/
CHOSEN_COACH_RANK/SCORING_WEIGHT/BENCH_WEIGHT (optimizer/pick_lineup_ilp.py),
and BUDGET (optimizer/pick_team_ilp.py) -- nothing here is recomputed.

ONE DELIBERATE DIFFERENCE from pick_lineup_ilp.py's own __main__: player
prices come from THIS round's real quotations (prices_df, loaded by
run_price_step via load_prices_history) instead of load_prices()'s single
current-snapshot file -- that's the whole point of wiring load_prices_history
into the live flow. Coach pricing still goes through rank_coaches ->
load_coach_info -> load_prices() (the generic snapshot) -- an existing
limitation of coach_ranking.py, unchanged by this slice.

STEP 1: collect_season(season) used to be fetch_season.py's entire
module-level script body, hardcoded to SEASON=2023 and executed on import.
It's now a season-parameterized function (see that file's own docstring for
why) specifically so this pipeline could import and call it against 2025
without touching the de-dup/fetch/store logic itself, which is unchanged.

STEP 2, WHY A PLAIN IMPORT: fetch_positions.py already targets this
pipeline's working season (its own module-level SEASON=2025 already
matches -- unlike fetch_season.py, it needed no season-parameter fix) but
isn't wrapped in a callable function, so importing IT IS running it. It's
also not skip-based like the box-score collector -- by design (see that
file's own module docstring), it deletes and re-inserts the season's whole
positions table every run, since every game must be re-scanned regardless
to catch inconsistent positions across games. So "newly added vs skipped"
doesn't map cleanly onto this step; it reports a before/after row-count
diff instead, to show whether the refresh actually changed anything. Uses
importlib to (re)run the module body on every call to run_positions_step,
not just the first import in the process -- a plain `import
data_collection.fetch_positions` would silently no-op on a second call in
the same process (Python caches modules), which would break the "repeatable"
part of a per-round refresh routine.

STEP 3, WHY THERE'S NO DB WRITE TO REPORT: build_turn_lookup returns an
in-memory dict only -- turn_tags.py never had a persistence layer to begin
with (see that file's own module docstring). "Refreshing" here means
re-deriving the lookup from the live schedule (so a reschedule that
happened since the last run is picked up), and reporting its current
shape -- not diffing against stored state, since none exists.
"""

import importlib
import sqlite3
import sys

import pandas as pd

from backtest.simulate_round import build_predictions
from data_collection.fetch_season import collect_season
from data_collection.load_prices_history import load_prices_for_round, store_prices
from features.turn_tags import build_turn_lookup, load_regular_season_schedule
from optimizer.availability import load_unavailable
from optimizer.coach_ranking import SHORTLIST_SIZE, rank_coaches
from optimizer.pick_lineup_ilp import (
    BENCH_WEIGHT,
    CHOSEN_COACH_RANK,
    RISK_LAMBDA,
    SCORING_WEIGHT,
    TURN_COVERAGE_WEIGHT,
    solve_lineup,
)
from optimizer.pick_team import POSITION_REQUIREMENTS, attach_positions, get_round_season
from optimizer.pick_team_ilp import BUDGET
from optimizer.turn_join import attach_turns
from scoring.season_rules import get_rules

SEASON = 2025
DB_PATH = "data/euroleague.db"

# See module docstring's TESTING ON A PAST ROUND section: Season 2025 is
# fully played, so this stands in for "the upcoming round" during testing.
TARGET_ROUND = 32


def get_positions_row_count(season, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM positions WHERE Season = ?", (season,)).fetchone()[0]
    conn.close()
    return count


def get_round_gamecodes(season, round_number):
    """Every regular-season Gamecode for one round number, via the same
    schedule load features/turn_tags.py's build_turn_lookup already uses.
    Fails loudly on a round number with no scheduled games (typo, or a
    round outside the regular season -- e.g. playoffs) rather than
    silently handing an empty target to the prediction/optimizer steps."""
    schedule_df = load_regular_season_schedule(season)
    gamecodes = sorted(schedule_df.loc[schedule_df["gameday"] == round_number, "game"].tolist())
    if not gamecodes:
        raise ValueError(
            f"No Season {season} regular-season games found for round {round_number} "
            f"-- check TARGET_ROUND is a real regular-season round (1-38 for 2025)."
        )
    return gamecodes


def run_box_score_step(season):
    """Step 1: incremental box-score collection -- reports the actual
    new Gamecodes saved, from collect_season's own return value."""
    print(f"\n=== Step 1/6: box-score collector (Season {season}) ===")
    summary = collect_season(season)
    if summary["new_gamecodes"]:
        print(
            f"Newly added: {summary['games_saved']} game(s), Gamecodes "
            f"{sorted(summary['new_gamecodes'])}"
        )
    else:
        print(f"Newly added: 0 games -- all {summary['rounds_scanned']} rounds already stored")
    return summary


def run_positions_step(season):
    """Step 2: positions refresh -- full delete+reinsert every call (see
    module docstring), reported as a before/after row-count diff."""
    print(f"\n=== Step 2/6: positions collector (Season {season}) ===")
    before = get_positions_row_count(season)

    module_name = "data_collection.fetch_positions"
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)

    after = get_positions_row_count(season)
    print(
        f"Positions refresh is a full delete+reinsert every run, not "
        f"skip-based -- before: {before} rows, after: {after} rows "
        f"({'changed' if before != after else 'unchanged'})"
    )
    return {"before": before, "after": after}


def run_turn_tag_step(season):
    """Step 3: rebuild the turn lookup from the live schedule (no DB write --
    see module docstring). Reports the T1/T2/RESCHEDULED shape."""
    print(f"\n=== Step 3/6: turn-tag refresh (Season {season}) ===")
    turn_lookup = build_turn_lookup(season)
    turns = list(turn_lookup.values())
    t1, t2, rescheduled = turns.count("T1"), turns.count("T2"), turns.count("RESCHEDULED")
    print(
        f"Rebuilt from the live schedule (in-memory only, no DB write): "
        f"{len(turn_lookup)} games tagged (T1={t1}, T2={t2}, RESCHEDULED={rescheduled})"
    )
    return turn_lookup


def status_summary(season, db_path=DB_PATH):
    """Latest round/gamecode present, total games, and the next/upcoming
    round -- the first round with any game missing from the DB, i.e. the
    one a manager still needs to pick a team for."""
    print(f"\n=== Status summary (Season {season}) ===")
    schedule_df = load_regular_season_schedule(season)
    round_to_gamecodes = schedule_df.groupby("gameday")["game"].apply(set).to_dict()

    conn = sqlite3.connect(db_path)
    present_gamecodes = set(
        pd.read_sql_query(
            "SELECT DISTINCT Gamecode FROM boxscores WHERE Season = ?", conn, params=[season]
        )["Gamecode"]
    )
    conn.close()

    total_games = len(present_gamecodes)
    latest_gamecode = max(present_gamecodes) if present_gamecodes else None
    latest_round_with_any_data = None
    if latest_gamecode is not None:
        match = schedule_df.loc[schedule_df["game"] == latest_gamecode, "gameday"]
        latest_round_with_any_data = int(match.iloc[0]) if len(match) else None

    all_rounds = sorted(round_to_gamecodes)
    complete_rounds = [r for r in all_rounds if round_to_gamecodes[r] <= present_gamecodes]
    incomplete_rounds = [r for r in all_rounds if r not in complete_rounds]

    latest_complete_round = max(complete_rounds) if complete_rounds else None
    next_round = min(incomplete_rounds) if incomplete_rounds else None

    print(f"Total games in DB: {total_games} / {len(schedule_df)} scheduled regular-season games")
    print(f"Latest Gamecode present: {latest_gamecode} (round {latest_round_with_any_data})")
    print(f"Latest COMPLETE round in DB: {latest_complete_round}")
    if next_round is not None:
        missing = sorted(round_to_gamecodes[next_round] - present_gamecodes)
        print(
            f"Next/upcoming round (to pick a team for): {next_round} -- "
            f"{len(missing)} game(s) not yet in DB: {missing}"
        )
    else:
        print("Next/upcoming round: none -- every scheduled regular-season round is fully collected")

    # --- MANUAL STEP: prices have no API (login-gated), so this can only
    # --- name where the file belongs, not fetch it -- see module docstring's
    # --- "WHERE THE MANUAL PRICE STEP BELONGS" section ---
    print("\nManual step before running the optimizer:")
    if next_round is not None:
        print(
            f"  Download round {next_round}'s price Excel from the EuroLeague Fantasy "
            f"site and place it at data/prices/round_{next_round:02d}/players_stats.xlsx, "
            f"then run `python -m data_collection.load_prices_history` to load it into "
            f"the prices table (data_collection/load_prices_history.py)."
        )
    else:
        print("  N/A -- no upcoming round to price (every scheduled round is fully collected).")

    return {
        "total_games": total_games,
        "latest_gamecode": latest_gamecode,
        "latest_complete_round": latest_complete_round,
        "next_round": next_round,
    }


def run_price_step(round_number, season=SEASON):
    """Step 4: load this round's real fantasy prices from
    data/prices/round_NN/players_stats.xlsx into the `prices` table --
    reuses load_prices_for_round (parsing + Season/Round tagging) and
    store_prices (skip-based on (Season, Round, Player_ID), safe to rerun)
    from data_collection/load_prices_history.py; nothing about the parsing
    or de-dup guard is redone here."""
    print(f"\n=== Step 4/6: price loader (Season {season}, Round {round_number}) ===")
    prices_df, coverage = load_prices_for_round(round_number, season=season)
    newly_stored = store_prices(prices_df)
    print(f"Source: {coverage['path']}")
    print(f"Usable prices parsed: {coverage['usable']}")
    print(
        f"Excluded -- head coaches: {coverage['head_coaches_excluded']}, "
        f"missing position: {coverage['missing_position_excluded']}, "
        f"placeholder-ID known duplicates: {coverage['placeholder_id_known_duplicates']}"
    )
    for row in coverage["placeholder_id_unrecovered_names"]:
        print(f"  unrecovered: {row['Name']} {row['Surname']} ({row['Team']})")
    print(f"Stored: {newly_stored} new / {coverage['usable'] - newly_stored} already in DB")
    return prices_df


def run_prediction_step(target_gamecodes):
    """Step 5: leakage-safe feature pipeline + prediction for the target
    round -- reuses build_predictions (backtest/simulate_round.py), which
    itself reuses add_rolling_form/add_rolling_minutes/add_team_rolling_defense
    (features/*.py) and trains the same LinearRegression-over-model_v3's-
    feature-set model_v3.py does, just restricted to Season 2025 games
    strictly before target_gamecodes rather than model_v3's fixed
    CUTOFF_GAMECODE split. Nothing about features or the model is
    reimplemented here."""
    print(
        f"\n=== Step 5/6: leakage-safe prediction "
        f"(Gamecodes {min(target_gamecodes)}-{max(target_gamecodes)}) ==="
    )
    model, history_computed, round_features, round_df = build_predictions(target_gamecodes)

    prior_games = history_computed["Gamecode"].nunique()
    round_players = round_df["Player_ID"].nunique()
    print(f"Trained on {prior_games} prior Season {SEASON} game(s) (strictly before this round)")
    print(
        f"{len(round_features)}/{round_players} round candidates have enough prior "
        f"history for a prediction ({round_players - len(round_features)} dropped, "
        f"e.g. season debuts)"
    )
    return round_features


def run_team_generation_step(round_features, target_gamecodes, prices_df, season=SEASON):
    """Step 6: the live optimizer -- attach positions/turns/this round's
    real prices, drop unavailable/unpriced candidates, build the coach
    shortlist, and solve the risk-aware turn-coverage-bonused lineup MILP.
    Reuses attach_positions/get_round_season (optimizer/pick_team.py),
    load_unavailable (optimizer/availability.py), build_turn_lookup
    (features/turn_tags.py) + attach_turns (optimizer/turn_join.py),
    rank_coaches/SHORTLIST_SIZE (optimizer/coach_ranking.py), get_rules
    (scoring/season_rules.py), and solve_lineup + RISK_LAMBDA/
    TURN_COVERAGE_WEIGHT/CHOSEN_COACH_RANK/SCORING_WEIGHT/BENCH_WEIGHT
    (optimizer/pick_lineup_ilp.py) -- exactly the constraints/objective
    that file's own __main__ uses. See the module docstring's ONE
    DELIBERATE DIFFERENCE note for the one thing that IS different: prices
    come from prices_df (this round's real quotations), not load_prices()'s
    generic snapshot."""
    print(f"\n=== Step 6/6: live optimizer (Season {season}) ===")

    df_round = attach_positions(round_features)

    unavailable = load_unavailable()
    excluded_ids = {entry["player_id"] for entry in unavailable}
    excluded_mask = df_round["Player_ID"].str.strip().isin(excluded_ids)
    excluded_names = df_round.loc[excluded_mask, "Player"].unique().tolist()
    df_round = df_round[~excluded_mask].copy()
    print(f"Excluded {len(excluded_names)} unavailable player(s) (data/unavailable.csv): {excluded_names}")

    price_lookup = dict(zip(prices_df["Player_ID"], prices_df["Quotation"]))
    df_round["price"] = df_round["Player_ID"].str.strip().map(price_lookup)
    no_price_mask = df_round["price"].isna()
    no_price_names = df_round.loc[no_price_mask, "Player"].unique().tolist()
    df_round = df_round[~no_price_mask].copy()
    print(f"Excluded {len(no_price_names)} player(s) with no matching price this round: {no_price_names}")

    season_resolved = get_round_season(df_round)
    captain_multiplier = get_rules(season_resolved)["captain_multiplier"]

    turn_lookup = build_turn_lookup(season_resolved)
    df_round = attach_turns(df_round, turn_lookup)
    flagged_turns = df_round[df_round["turn"].isin(["RESCHEDULED", "NO_TURN_FOUND"])]
    if len(flagged_turns):
        print(
            f"{flagged_turns['Player'].nunique()} candidate(s) have no clean T1/T2 turn "
            f"(excluded from the turn-coverage bonus, not from the roster pool)"
        )

    # --- coach shortlist (same ranking coach_ranking.py prints standalone) ---
    ranked_coaches = rank_coaches(target_gamecodes)
    shortlist = ranked_coaches.head(SHORTLIST_SIZE).reset_index(drop=True)
    shortlist_display = shortlist.copy()
    shortlist_display.insert(0, "rank", shortlist_display.index + 1)
    print(f"\nCoach shortlist for this round (top {SHORTLIST_SIZE} by recent-margin):")
    print(
        shortlist_display[
            ["rank", "coach", "Team", "recent_avg_margin", "opponent", "coach_price"]
        ].to_string(index=False)
    )

    if not isinstance(CHOSEN_COACH_RANK, int) or not (1 <= CHOSEN_COACH_RANK <= SHORTLIST_SIZE):
        raise ValueError(
            f"CHOSEN_COACH_RANK must be an integer between 1 and {SHORTLIST_SIZE}, "
            f"got {CHOSEN_COACH_RANK!r}"
        )
    chosen_coach = shortlist.iloc[CHOSEN_COACH_RANK - 1]
    coach_name = chosen_coach["coach"]
    coach_team = chosen_coach["Team"]
    coach_price = chosen_coach["coach_price"]
    if pd.isna(coach_price):
        raise ValueError(
            f"Chosen coach {coach_name} ({coach_team}) has no price in the price sheet."
        )
    player_budget = BUDGET - coach_price
    print(
        f"\nChosen coach (shortlist rank {CHOSEN_COACH_RANK}/{SHORTLIST_SIZE}): "
        f"{coach_name} ({coach_team}) -- price {coach_price:.1f}, "
        f"player budget {BUDGET} - {coach_price:.1f} = {player_budget:.1f}"
    )

    missing_consistency = df_round["consistency_5"].isna()
    if missing_consistency.any():
        raise ValueError(
            f"{missing_consistency.sum()} candidate(s) have no consistency_5, which "
            f"should be impossible -- build_predictions' own dropna(subset=FEATURE_COLS) "
            f"must have a bug: {df_round.loc[missing_consistency, 'Player'].tolist()}"
        )
    df_round["objective_score"] = df_round["predicted_score"] - RISK_LAMBDA * df_round["consistency_5"]

    result = solve_lineup(
        df_round,
        "objective_score",
        captain_multiplier,
        budget=player_budget,
        turn_coverage_weight=TURN_COVERAGE_WEIGHT,
    )
    starters_idx, sixth_idx, bench_idx = result["starters_idx"], result["sixth_idx"], result["bench_idx"]
    captain_idx, roster_idx = result["captain_idx"], result["roster_idx"]

    print(f"\nSolver status: {result['status']}")
    print(f"Formation: {result['formation']}")
    print(f"Season: {season_resolved}, captain_multiplier: {captain_multiplier}, RISK_LAMBDA: {RISK_LAMBDA}")

    role_by_idx = {}
    for idx in starters_idx:
        role_by_idx[idx] = "Starter"
    for idx in sixth_idx:
        role_by_idx[idx] = "6th Man"
    for idx in bench_idx:
        role_by_idx[idx] = "Bench"
    role_order = {"Coach": 0, "Starter": 1, "6th Man": 2, "Bench": 3}

    players_table = df_round.loc[roster_idx].copy()
    players_table["role"] = players_table.index.map(role_by_idx)
    players_table["Player"] = players_table["Player"] + players_table.index.to_series().apply(
        lambda idx: " (C)" if idx == captain_idx else ""
    )

    coach_row = pd.DataFrame(
        [
            {
                "Player": f"{coach_name} ({coach_team})",
                "role": "Coach",
                "position": "-",
                "turn": "-",
                "price": coach_price,
                "predicted_score": float("nan"),
                "consistency_5": float("nan"),
                "objective_score": float("nan"),
            }
        ]
    )

    columns = [
        "Player",
        "role",
        "position",
        "turn",
        "price",
        "predicted_score",
        "consistency_5",
        "objective_score",
    ]
    full_team = pd.concat([coach_row, players_table[columns]], ignore_index=True)
    full_team = full_team.sort_values(by="role", key=lambda col: col.map(role_order))

    print(f"\nFull 11-slot recommended team (Round {min(target_gamecodes)}-{max(target_gamecodes)}):")
    print(full_team[columns].to_string(index=False))

    coverage_rows = []
    for position in POSITION_REQUIREMENTS:
        group = df_round.loc[roster_idx]
        group = group[group["position"] == position]
        coverage_rows.append(
            {
                "position": position,
                "rostered": len(group),
                "T1_count": (group["turn"] == "T1").sum(),
                "T2_count": (group["turn"] == "T2").sum(),
                "both_turns_covered": result["turn_coverage"].get(position),
            }
        )
    print(f"\nTurn coverage by position (TURN_COVERAGE_WEIGHT={TURN_COVERAGE_WEIGHT}):")
    print(pd.DataFrame(coverage_rows).to_string(index=False))

    total_quotation = df_round.loc[roster_idx, "price"].sum()
    print(f"\nPlayer quotation used: {total_quotation:.1f} / {player_budget:.1f}")
    print(f"Total squad cost (coach + players): {coach_price + total_quotation:.1f} / {BUDGET}")

    return {
        "full_team": full_team,
        "shortlist": shortlist,
        "result": result,
    }


if __name__ == "__main__":
    print(f"Data refresh: Season {SEASON}")
    run_box_score_step(SEASON)
    run_positions_step(SEASON)
    run_turn_tag_step(SEASON)
    status_summary(SEASON)

    print(f"\nTeam generation: Season {SEASON}, targeting Round {TARGET_ROUND} "
          f"(TEST -- see module docstring's TESTING ON A PAST ROUND section)")
    target_gamecodes = get_round_gamecodes(SEASON, TARGET_ROUND)
    prices_df = run_price_step(TARGET_ROUND, SEASON)
    round_features = run_prediction_step(target_gamecodes)
    run_team_generation_step(round_features, target_gamecodes, prices_df, SEASON)
