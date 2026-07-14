"""Per-team rolling defensive rating: fantasy points conceded to opponents.

Reuses base_fantasy_score from scoring/base_score.py so the score formula
is never duplicated here. Rolling averages are shifted so a team's own
current game is never included in its own defensive rating (no data
leakage) -- only strictly earlier games in that team's chronological
history are used, matching the approach in features/minutes.py and
features/rolling_form.py.
"""

import sqlite3

import pandas as pd

from scoring.base_score import base_fantasy_score


def add_team_rolling_defense(df):
    df["base_score"] = df.apply(base_fantasy_score, axis=1)

    team_game = df.groupby(["Season", "Gamecode", "Team"])["base_score"].sum().reset_index()
    team_game = team_game.rename(columns={"base_score": "team_score"})

    matchup = team_game.merge(team_game, on=["Season", "Gamecode"], suffixes=("", "_opp"))
    matchup = matchup[matchup["Team"] != matchup["Team_opp"]]
    matchup = matchup.rename(columns={"team_score_opp": "conceded"})
    matchup = matchup[["Season", "Gamecode", "Team", "conceded"]]

    matchup = matchup.sort_values(["Team", "Season", "Gamecode"]).reset_index(drop=True)

    by_team = matchup.groupby("Team")["conceded"]
    matchup["rolling_5_conceded"] = by_team.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    return matchup


if __name__ == "__main__":
    conn = sqlite3.connect("data/euroleague.db")
    df = pd.read_sql_query(
        "SELECT * FROM boxscores WHERE Player_ID NOT IN ('Team', 'Total')",
        conn,
    )
    conn.close()

    matchup = add_team_rolling_defense(df)

    sample = matchup[matchup["Team"] == "MUN"]
    print(
        sample[["Season", "Gamecode", "Team", "conceded", "rolling_5_conceded"]].to_string(
            index=False
        )
    )
