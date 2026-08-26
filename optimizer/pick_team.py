"""First optimizer skeleton: pick the 10-player team that maximizes total
predicted fantasy score for one round, subject only to roster shape
(4 Guards, 4 Forwards, 2 Centers). No budget, club limit, formation, or
captain yet -- those come later.

Reuses train_and_predict from prediction/model_v3.py instead of
retraining or recomputing predictions here. Only test-set (out-of-sample)
predictions are used -- picking a team from train-set predictions would
mean optimizing over games the model already memorized while fitting,
which isn't a fair stand-in for real deployment on unseen games.

Positions come from the real `positions` table in data/euroleague.db
(built by data_collection/fetch_positions.py from GameStats.get_game_stats,
100% coverage for 2023), joined on Player_ID. attach_positions fails loudly
if any candidate player has no match -- with full coverage, a miss can only
be a join bug (wrong season, bad normalization, stale DB), never a real gap.
"""

import sqlite3

import pandas as pd

from prediction.model_v3 import train_and_predict

ROUND_GAMECODES = range(253, 262)  # one verified full round: 9 games, 18 teams, in the test set

# One real, complete Season 2025 round: 10 consecutive Gamecodes, 20 distinct
# teams with no repeats (verified against boxscores), fully inside
# model_v3's test range (Gamecode >= CUTOFF_GAMECODE = 304). Unlike
# ROUND_GAMECODES above (calibrated to the 2023 test set's 9-games/18-teams
# round shape), 2025 has 20 teams, so a round is 10 games. Lives here
# (rather than in pick_lineup_ilp.py, where it used to be) so both
# pick_lineup_ilp.py and coach_ranking.py can import it without an import
# cycle between those two files.
ROUND_GAMECODES_2025 = range(311, 321)

POSITION_REQUIREMENTS = {"G": 4, "F": 4, "C": 2}


def get_round_season(df_round):
    seasons = df_round["Season"].unique()
    assert len(seasons) == 1, "Expected one round to span exactly one season"
    return int(seasons[0])


def load_positions(season):
    conn = sqlite3.connect("data/euroleague.db")
    positions_df = pd.read_sql_query(
        "SELECT Player_ID, Position FROM positions WHERE Season = ?", conn, params=[season]
    )
    conn.close()
    return dict(zip(positions_df["Player_ID"], positions_df["Position"]))


def attach_positions(df_round):
    season = get_round_season(df_round)
    positions_lookup = load_positions(season)

    df_round = df_round.copy()
    normalized_id = df_round["Player_ID"].str.strip().str.lstrip("P")
    df_round["position"] = normalized_id.map(positions_lookup)

    missing = df_round[df_round["position"].isna()]
    if len(missing):
        raise ValueError(
            f"{len(missing)} candidate players have no position in the positions "
            f"table for season {season} (2023 coverage is 100%, so this means a "
            f"join bug, not a real data gap): {missing['Player'].tolist()}"
        )
    return df_round


if __name__ == "__main__":
    model, feature_cols, train, test = train_and_predict()

    df_round = test[test["Gamecode"].isin(ROUND_GAMECODES)].copy()
    df_round = attach_positions(df_round)

    selected_groups = []
    for position, count in POSITION_REQUIREMENTS.items():
        pool = df_round[df_round["position"] == position]
        if len(pool) < count:
            raise ValueError(
                f"Not enough players at position {position} in this round: "
                f"need {count}, have {len(pool)}"
            )
        selected_groups.append(pool.nlargest(count, "predicted_score"))

    team = pd.concat(selected_groups)

    print(team[["Player", "position", "predicted_score"]].to_string(index=False))
    print(f"\nTotal predicted score: {team['predicted_score'].sum():.2f}")
