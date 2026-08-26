"""Leakage-safe single-round backtest for Season 2025.

The live optimizer (optimizer/pick_lineup_ilp.py) always predicts an
UNPLAYED round -- there's no "actual score" to check it against yet. This
file answers a different question: if the engine had picked a team for a
round that HAS already been played, using only the information available
before that round started, how many points would it actually have scored,
compared to two simple baselines? It's a dry run of the whole pipeline
against ground truth. No DB writes -- read-only, report only.

STRICT LEAKAGE RULE: the engine may only use Season 2025 games with
Gamecode strictly before the target round to train the model and compute
every rolling feature. The target round's own box-score stats are never
merely "unused" -- they are structurally ABSENT from the data fed into
feature computation and training. add_rolling_form/add_rolling_minutes/
add_team_rolling_defense already exclude a row's own game from its own
features via shift(1) (hand-verified in earlier slices, see features/*.py),
so relying on that alone would already be leakage-free. This file goes
further as a belt-and-suspenders measure: every box-score stat column for
the target round's rows is overwritten with 0 (Minutes -> "DNP") before
those rows are combined with real history and run through the feature
pipeline. Only roster metadata survives (Player_ID, Player, Team, Gamecode,
Season) -- exactly what a manager knows before tip-off. This way, even a
hypothetical bug in the rolling/shift logic could not leak real target-round
results, because those results are never present anywhere in the frame the
model and features are computed from. The round's REAL box scores are
loaded completely separately (round_df, never touched by feature/model
code) and used only AFTER a lineup is selected, to compute ACTUAL points.

Target round: Season 2025, Gamecode 231-240 -- one real, complete round (10
games, 20 distinct teams, no repeats, verified against boxscores).
Mid-to-late season (230 prior Season-2025 games precede it) so every
candidate has a real rolling history rather than a cold start.

Reuses add_rolling_form (features/rolling_form.py), add_rolling_minutes
(features/minutes.py), add_team_rolling_defense (features/matchup.py),
base_fantasy_score (scoring/base_score.py), get_rules
(scoring/season_rules.py), attach_positions/get_round_season
(optimizer/pick_team.py), solve_lineup/SCORING_WEIGHT/BENCH_WEIGHT (which
already carry the roster-shape/budget/club-cap constraints and their
BUDGET/MAX_PER_CLUB defaults -- optimizer/pick_lineup_ilp.py), and
load_prices (data_collection/load_prices.py) -- nothing here is recomputed.

data/unavailable.csv (hand-edited out/doubtful list) is deliberately NOT
consulted: it represents current manual overrides for an upcoming round,
which has no meaning when reproducing a round that already happened.

The official round name/number (e.g. "Regular Season, Round 24") is looked
up from euroleague_api's Schedule endpoint -- the same source
data_collection/fetch_season.py already uses -- purely for display; nothing
about the leakage-safe pipeline depends on it. This is the only network
call in the file; everything else reads from the local DB.
"""

import sqlite3

import numpy as np
import pandas as pd
from euroleague_api.schedule import Schedule
from sklearn.linear_model import LinearRegression

from data_collection.load_prices import load_prices
from features.matchup import add_team_rolling_defense
from features.minutes import add_rolling_minutes
from features.rolling_form import add_rolling_form
from optimizer.pick_lineup_ilp import BENCH_WEIGHT, SCORING_WEIGHT, solve_lineup
from optimizer.pick_team import attach_positions, get_round_season
from scoring.base_score import base_fantasy_score
from scoring.season_rules import get_rules

DB_PATH = "data/euroleague.db"
SEASON = 2025
TARGET_ROUND_GAMECODES = range(231, 241)
TARGET_START = min(TARGET_ROUND_GAMECODES)

FEATURE_COLS = [
    "rolling_5",
    "rolling_3",
    "rolling_5_minutes",
    "rolling_3_minutes",
    "consistency_5",
    "opp_rolling_5_conceded",
    "distance_from_norm",
]

# Every box-score stat column that feeds base_fantasy_score or a rolling
# feature, blanked out on the target round's rows before feature/model
# computation -- see the module docstring's STRICT LEAKAGE RULE.
STAT_COLUMNS_TO_BLIND = [
    "Points",
    "FieldGoalsMade2",
    "FieldGoalsAttempted2",
    "FieldGoalsMade3",
    "FieldGoalsAttempted3",
    "FreeThrowsMade",
    "FreeThrowsAttempted",
    "OffensiveRebounds",
    "DefensiveRebounds",
    "TotalRebounds",
    "Assistances",
    "Steals",
    "Turnovers",
    "BlocksFavour",
    "BlocksAgainst",
    "FoulsCommited",
    "FoulsReceived",
    "Valuation",
    "Plusminus",
]


ROUND_TYPE_LABELS = {"RS": "Regular Season", "PO": "Playoffs", "FF": "Final Four"}


def get_round_name(season, gamecodes):
    """Official EuroLeague round name/number for a set of Gamecodes, via
    euroleague_api's Schedule (same source fetch_season.py uses). Display
    only -- not used anywhere in the leakage-safe pipeline above."""
    schedule_df = Schedule().get_schedule(season)
    schedule_df["game"] = schedule_df["game"].astype(int)
    round_rows = schedule_df[schedule_df["game"].isin(gamecodes)]

    gamedays = round_rows["gameday"].unique()
    round_types = round_rows["round"].unique()
    assert len(gamedays) == 1, f"Target round spans multiple gamedays: {gamedays}"
    assert len(round_types) == 1, f"Target round spans multiple round types: {round_types}"

    round_type_label = ROUND_TYPE_LABELS.get(round_types[0], round_types[0])
    dates = sorted(round_rows["date"].unique())
    date_range = dates[0] if len(dates) == 1 else f"{dates[0]} - {dates[-1]}"
    return f"{round_type_label}, Round {gamedays[0]} ({date_range})"


def load_season_boxscores():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM boxscores WHERE Season = ? AND Player_ID NOT IN ('Team', 'Total')",
        conn,
        params=[SEASON],
    )
    conn.close()
    return df


def build_predictions(target_gamecodes, boxscores=None):
    """Train on, and compute every feature from, ONLY Season 2025 games
    strictly before target_gamecodes. Returns (model, history_computed,
    round_features, round_df):
      - history_computed: every pre-round row with base_score/features
        attached, UNFILTERED for feature completeness -- e.g. a player's
        3rd-ever game has a real base_score here even though its own
        consistency_5 is still NaN (not enough games yet for a std-dev).
        Used for the naive baseline's prior-average, which only needs
        base_score, not the rolling features.
      - round_features: target-round candidates with predicted_score,
        computed with zero visibility into their real outcome. Filtered to
        rows where every FEATURE_COLS value is present, since those feed
        the model directly.
      - round_df: the target round's REAL box scores -- untouched by any
        feature/model code, returned only so the caller can compute actual
        points AFTER a lineup is picked.

    target_gamecodes is any iterable of Gamecodes making up one round (see
    TARGET_ROUND_GAMECODES for the shape). boxscores lets a caller that
    runs this across many rounds (backtest/simulate_many.py) load the full
    season once and pass it in, instead of re-querying the DB per round --
    the leakage-safe logic itself is unchanged either way.
    """
    target_start = min(target_gamecodes)
    if boxscores is None:
        boxscores = load_season_boxscores()

    history_df = boxscores[boxscores["Gamecode"] < target_start].copy()
    round_df = boxscores[boxscores["Gamecode"].isin(target_gamecodes)].copy()

    round_blinded = round_df.copy()
    round_blinded[STAT_COLUMNS_TO_BLIND] = 0
    round_blinded["Minutes"] = "DNP"

    combined = pd.concat([history_df, round_blinded], ignore_index=True)
    combined = add_rolling_form(combined)
    combined = add_rolling_minutes(combined)

    team_defense = add_team_rolling_defense(combined)
    opp_defense = team_defense.merge(
        team_defense, on=["Season", "Gamecode"], suffixes=("", "_opp")
    )
    opp_defense = opp_defense[opp_defense["Team"] != opp_defense["Team_opp"]]
    opp_defense = opp_defense[["Season", "Gamecode", "Team", "rolling_5_conceded_opp"]]
    opp_defense = opp_defense.rename(columns={"rolling_5_conceded_opp": "opp_rolling_5_conceded"})
    combined = combined.merge(opp_defense, on=["Season", "Gamecode", "Team"], how="left")

    history_computed = combined[combined["Gamecode"] < target_start]
    train_df = history_computed.dropna(subset=FEATURE_COLS)
    round_features = combined[combined["Gamecode"].isin(target_gamecodes)].dropna(
        subset=FEATURE_COLS
    )

    # Belt-and-suspenders: prove the blinding actually held, not just assume it.
    assert (train_df["Gamecode"] < target_start).all(), "Train set leaked target-round games"
    assert round_blinded[STAT_COLUMNS_TO_BLIND].abs().sum().sum() == 0, "Blinding failed"

    model = LinearRegression()
    model.fit(train_df[FEATURE_COLS], train_df["base_score"])

    round_features = round_features.copy()
    round_features["predicted_score"] = model.predict(round_features[FEATURE_COLS])

    return model, history_computed, round_features, round_df


def score_team(df_round, result, captain_multiplier, score_col):
    """Apply the real lineup weighting (100% starters+6th man, 50% bench,
    captain's extra share) to score_col for a solved lineup."""
    starters_idx = result["starters_idx"]
    sixth_idx = result["sixth_idx"]
    bench_idx = result["bench_idx"]
    captain_idx = result["captain_idx"]
    captain_score = df_round.loc[captain_idx, score_col]
    return (
        df_round.loc[starters_idx + sixth_idx, score_col].sum() * SCORING_WEIGHT
        + df_round.loc[bench_idx, score_col].sum() * BENCH_WEIGHT
        + (captain_multiplier - 1) * captain_score
    )


def assert_roster_complete(label, df_round, result):
    """Fail loudly if any team selected a candidate with no price or
    position -- both should already be impossible (df_round is pre-filtered
    to priced candidates, and attach_positions() already raises pool-wide
    for any missing position), so this only fires if an upstream guarantee
    is ever weakened."""
    roster_idx = result["roster_idx"]
    missing_price = df_round.loc[roster_idx, "price"].isna()
    if missing_price.any():
        raise ValueError(
            f"{label} team selected candidate(s) with no price: "
            f"{df_round.loc[roster_idx][missing_price]['Player'].tolist()}"
        )
    missing_position = df_round.loc[roster_idx, "position"].isna()
    if missing_position.any():
        raise ValueError(
            f"{label} team selected candidate(s) with no position: "
            f"{df_round.loc[roster_idx][missing_position]['Player'].tolist()}"
        )
    missing_actual = df_round.loc[roster_idx, "actual_score"].isna()
    if missing_actual.any():
        raise ValueError(
            f"{label} team selected candidate(s) with no actual score for "
            f"the target round: {df_round.loc[roster_idx][missing_actual]['Player'].tolist()}"
        )


if __name__ == "__main__":
    print(
        f"Target round: Season {SEASON}, Gamecode {TARGET_ROUND_GAMECODES.start}-"
        f"{TARGET_ROUND_GAMECODES.stop - 1}"
    )
    print(f"EuroLeague round: {get_round_name(SEASON, TARGET_ROUND_GAMECODES)}")

    model, history_computed, round_features, round_df = build_predictions(TARGET_ROUND_GAMECODES)

    prior_game_count = history_computed["Gamecode"].nunique()
    print(
        f"Training/feature history: Season {SEASON}, Gamecode < {TARGET_START} "
        f"({prior_game_count} games)\n"
    )

    round_player_count = round_df["Player_ID"].nunique()
    print(
        f"Round candidates: {round_player_count} players played; "
        f"{len(round_features)} have enough prior 2025 history for a prediction "
        f"({round_player_count - len(round_features)} dropped, e.g. season debuts)\n"
    )

    df_round = attach_positions(round_features)

    # --- join real quotations, exclude candidates with no matching price
    # --- from the pool entirely, same treatment as optimizer/pick_lineup_ilp.py ---
    prices_df, _ = load_prices()
    price_lookup = dict(zip(prices_df["Player_ID"], prices_df["Quotation"]))
    df_round["price"] = df_round["Player_ID"].str.strip().map(price_lookup)

    no_price_mask = df_round["price"].isna()
    no_price_names = df_round.loc[no_price_mask, "Player"].unique().tolist()
    df_round = df_round[~no_price_mask].copy()

    print(f"Excluded {len(no_price_names)} player(s) with no matching price from the candidate pool:")
    for name in no_price_names:
        print(f"  {name}")

    # --- naive baseline score: each candidate's own Season-2025 average
    # --- base_score across all real games strictly before the target round.
    # --- Uses history_computed (unfiltered for rolling-feature completeness)
    # --- rather than the model's train set, since an early-career game can
    # --- have a perfectly real base_score while still lacking e.g.
    # --- consistency_5 (not enough games yet for a std-dev) ---
    prior_avg_lookup = (
        history_computed.assign(pid=history_computed["Player_ID"].str.strip())
        .groupby("pid")["base_score"]
        .mean()
    )
    df_round["prior_avg_score"] = df_round["Player_ID"].str.strip().map(prior_avg_lookup)
    missing_prior_avg = df_round["prior_avg_score"].isna()
    if missing_prior_avg.any():
        raise ValueError(
            f"Candidate(s) with a predicted_score but no prior-average score, "
            f"which should be impossible: {df_round.loc[missing_prior_avg, 'Player'].tolist()}"
        )

    # --- random baseline score: fixed seed so the "random" team is
    # --- reproducible across runs, not different every time ---
    rng = np.random.default_rng(42)
    df_round["random_score"] = rng.random(len(df_round))

    # --- actual points: computed from round_df (real, unblinded box scores),
    # --- joined onto df_round only NOW, after every team below is selected
    # --- using nothing but predicted_score / prior_avg_score / random_score ---
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

    for label, result in [("Engine", engine_result), ("Naive", naive_result), ("Random", random_result)]:
        assert_roster_complete(label, df_round, result)

    # --- engine team: one combined table (role, captain, position, price,
    # --- predicted vs actual, and their difference so the model's misses
    # --- are easy to eyeball) instead of three separate blocks ---
    role_by_idx = {}
    for idx in engine_result["starters_idx"]:
        role_by_idx[idx] = "starter"
    for idx in engine_result["sixth_idx"]:
        role_by_idx[idx] = "6th man"
    for idx in engine_result["bench_idx"]:
        role_by_idx[idx] = "bench"
    role_order = {"starter": 0, "6th man": 1, "bench": 2}

    engine_table = df_round.loc[engine_result["roster_idx"]].copy()
    engine_table["role"] = engine_table.index.map(role_by_idx)
    engine_table["captain"] = engine_table.index == engine_result["captain_idx"]
    engine_table["diff (predicted - actual)"] = (
        engine_table["predicted_score"] - engine_table["actual_score"]
    )
    engine_table = engine_table.sort_values(
        by=["role", "predicted_score"],
        key=lambda col: col.map(role_order) if col.name == "role" else col,
        ascending=[True, False],
    )

    columns = [
        "Player",
        "role",
        "captain",
        "position",
        "price",
        "predicted_score",
        "actual_score",
        "diff (predicted - actual)",
    ]

    print(f"\nEngine solver status: {engine_result['status']}")
    print(f"Engine formation: {engine_result['formation']}")
    print(f"Season: {season}, captain_multiplier: {captain_multiplier}")

    print("\nEngine lineup:")
    print(engine_table[columns].to_string(index=False))

    # --- score all three teams on ACTUAL points ---
    engine_actual = score_team(df_round, engine_result, captain_multiplier, "actual_score")
    naive_actual = score_team(df_round, naive_result, captain_multiplier, "actual_score")
    random_actual = score_team(df_round, random_result, captain_multiplier, "actual_score")

    engine_predicted = score_team(df_round, engine_result, captain_multiplier, "predicted_score")

    print(f"\nEngine weighted predicted total (pre-round estimate): {engine_predicted:.2f}")

    print(f"\n{'Team':<10}{'Formation':>12}{'Actual points':>16}")
    print(f"{'Engine':<10}{engine_result['formation']:>12}{engine_actual:>16.2f}")
    print(f"{'Naive':<10}{naive_result['formation']:>12}{naive_actual:>16.2f}")
    print(f"{'Random':<10}{random_result['formation']:>12}{random_actual:>16.2f}")
