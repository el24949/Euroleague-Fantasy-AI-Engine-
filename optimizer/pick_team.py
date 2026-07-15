"""First optimizer skeleton: pick the 10-player team that maximizes total
predicted fantasy score for one round, subject only to roster shape
(4 Guards, 4 Forwards, 2 Centers). No budget, club limit, formation, or
captain yet -- those come later.

Reuses train_and_predict from prediction/model_v3.py instead of
retraining or recomputing predictions here. Only test-set (out-of-sample)
predictions are used -- picking a team from train-set predictions would
mean optimizing over games the model already memorized while fitting,
which isn't a fair stand-in for real deployment on unseen games.

PLACEHOLDER: real player positions (data requirement row 3) don't exist
yet. Positions below are a deterministic hash of Player_ID, purely so the
roster-shape constraint has something to optimize against. This has no
basketball meaning and must be replaced once real position data lands.
"""

import hashlib

import pandas as pd

from prediction.model_v3 import train_and_predict

ROUND_GAMECODES = range(253, 262)  # one verified full round: 9 games, 18 teams, in the test set
POSITION_REQUIREMENTS = {"G": 4, "F": 4, "C": 2}


def placeholder_position(player_id):
    digest = hashlib.md5(player_id.encode()).hexdigest()
    return ["G", "F", "C"][int(digest, 16) % 3]


if __name__ == "__main__":
    model, feature_cols, train, test = train_and_predict()

    df_round = test[test["Gamecode"].isin(ROUND_GAMECODES)].copy()
    df_round["position"] = df_round["Player_ID"].apply(placeholder_position)

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
