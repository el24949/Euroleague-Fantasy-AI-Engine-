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


def match_opponents(df, value_col):
    """Self-merge each (Season, Gamecode, Team)'s value_col total against
    its opponent's total in the same game. One row per team-game, with
    Team_opp (the opponent's code) and '{value_col}_opp' (the opponent's
    total) attached -- the "which team played which team, in which game"
    logic lives here exactly once, shared by add_team_rolling_defense
    (fantasy points conceded, below) and optimizer/coach_ranking.py (real
    point margin), instead of two separate self-merges.
    """
    team_game = df.groupby(["Season", "Gamecode", "Team"])[value_col].sum().reset_index()
    matchup = team_game.merge(team_game, on=["Season", "Gamecode"], suffixes=("", "_opp"))
    matchup = matchup[matchup["Team"] != matchup["Team_opp"]]
    return matchup[["Season", "Gamecode", "Team", "Team_opp", value_col, f"{value_col}_opp"]]


def add_team_rolling_defense(df):
    df["base_score"] = df.apply(base_fantasy_score, axis=1)

    matchup = match_opponents(df, "base_score")
    matchup = matchup.rename(columns={"base_score_opp": "conceded"})
    matchup = matchup[["Season", "Gamecode", "Team", "conceded"]]

    matchup = matchup.sort_values(["Team", "Season", "Gamecode"]).reset_index(drop=True)

    # Gamecode resets each season, so grouping on Team alone would let the
    # rolling window reach across the season boundary (see
    # features/rolling_form.py for the same fix). [Season, Team] gives each
    # club a fresh group per season.
    by_team = matchup.groupby(["Season", "Team"])["conceded"]
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
