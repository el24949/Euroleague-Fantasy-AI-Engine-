"""Assign each Season 2025 game a turn label (T1 or T2) within its round,
based on schedule dates -- i.e. which of a round's two playing days a game
falls on. Follows up on an earlier investigation into whether euroleague-api
exposes this directly (it doesn't -- no field names a turn/day-of-round;
see git history for that investigation). No DB writes yet --
build_turn_lookup() returns a dict in memory; this is a slice to verify the
clustering before deciding how, or whether, to persist it.

Reuses Schedule (euroleague_api.schedule -- the same source
data_collection/fetch_season.py and optimizer/coach_ranking.py's round-name
lookup already use). No boxscores/features function is recomputed here --
this is schedule-only information, independent of any player or team stat.

THE RESCHEDULE-OUTLIER PROBLEM: ~1/3 of Season 2025 regular-season rounds
(13 of 38) have 3+ distinct dates instead of a clean 2 -- but "more dates"
does NOT mean "more turns". Two different situations produce the extra
dates, and neither is a real T3:
  (a) a game gets postponed and replayed weeks or months later -- verified:
      round 14's Olympiacos-Fenerbahce moved from Dec 5 to Mar 17, a
      102-day jump; round 30 has two such outliers, one on Mar 17 and one
      on Apr 12;
  (b) a single game gets moved a day earlier or later than the round's
      main two dates for broadcast reasons, with no real postponement --
      e.g. round 3: 5 games on Oct 9, 4 on Oct 10, 1 lone game on Oct 8.
      Every gap here is only 1 day, nothing "far off", yet it's still not
      a genuine 3rd turn -- just one game shifted a day.

A pure date-GAP threshold can't tell these apart from a clean split by
itself. Round 33 is the clearest counter-example: 8 games on Mar 24, 1 lone
game on Mar 25 (1 day later), 1 lone game on Mar 31 (6 days after Mar 25,
7 after Mar 24) -- and 6 days is INSIDE the range of gaps seen in ordinary
international-break rounds (up to 6 days, e.g. round 18's Dec 23 -> Dec
26). A day-gap cutoff generous enough to keep round 18's legitimate 3-day
gap as one turn-transition would also swallow round 33's Mar 31 reschedule
into the same cluster as everything before it.

So the clustering here is GAME-COUNT-first, proximity only as a tiebreak:
the two dates carrying the most games become the T1/T2 core (earlier =
T1, later = T2); every other date is labelled RESCHEDULED rather than
guessed into T1 or T2, since the schedule data has no field for a game's
*originally intended* date (confirmeddate/confirmedtime are booleans, not
alternate dates -- confirmed in the earlier investigation). Ties for the
2nd core slot (e.g. round 30's Mar 5 vs Mar 6, both 4 games -- no tie to
break there since both trivially outrank the two 1-game dates; the real
tie case is round 33's Mar 25 vs Mar 31, both lone games) are broken by
picking whichever tied date is closer to the already-selected core date --
this is what correctly keeps round 33's Mar 25 as T2 while flagging Mar 31,
despite both being lone games with identical counts.
"""

import pandas as pd
from euroleague_api.schedule import Schedule

SEASON = 2025


def load_regular_season_schedule(season=SEASON):
    """Season's regular-season games only, with date parsed to a real
    Timestamp -- the raw 'date' column is a string like 'Jan 22, 2026'."""
    df = Schedule().get_schedule(season)
    df = df[df["round"] == "RS"].copy()
    df["game"] = df["game"].astype(int)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%b %d, %Y")
    return df


def assign_round_turns(round_games):
    """T1/T2/RESCHEDULED for one round's games (a DataFrame with 'game'
    and 'date_parsed' columns) -- see module docstring for the
    count-first, proximity-tiebreak algorithm. Returns {game: turn}."""
    date_counts = round_games["date_parsed"].value_counts()

    if len(date_counts) == 1:
        # Every game on one date -- no split to make. Not observed in
        # Season 2025 regular-season data, but handled rather than
        # crashing if a future season (or the playoffs/Final Four) has it.
        return {row["game"]: "T1" for _, row in round_games.iterrows()}

    sorted_counts = date_counts.sort_values(ascending=False)
    first_core_date = sorted_counts.index[0]

    if len(date_counts) == 2:
        second_core_date = sorted_counts.index[1]
    else:
        remaining = sorted_counts.iloc[1:]
        second_highest_count = remaining.iloc[0]
        tied = remaining[remaining == second_highest_count].index.tolist()
        if len(tied) == 1:
            second_core_date = tied[0]
        else:
            # Tiebreak: whichever tied date is fewest days from the
            # already-selected core date (see module docstring's round 33
            # example -- this is exactly what separates a 1-day-later lone
            # game from a genuinely postponed one).
            gaps = {d: abs((d - first_core_date).days) for d in tied}
            second_core_date = min(gaps, key=gaps.get)

    t1_date, t2_date = sorted([first_core_date, second_core_date])

    turns = {}
    for _, row in round_games.iterrows():
        if row["date_parsed"] == t1_date:
            turns[row["game"]] = "T1"
        elif row["date_parsed"] == t2_date:
            turns[row["game"]] = "T2"
        else:
            turns[row["game"]] = "RESCHEDULED"
    return turns


def build_turn_lookup(season=SEASON):
    """{(Season, Gamecode): turn} for every regular-season game. 'game'
    (not 'gamecode', which is a string like 'E2025_7') is the field that
    matches boxscores.Gamecode -- same field data_collection/fetch_season.py
    already uses for the same reason."""
    schedule_df = load_regular_season_schedule(season)

    lookup = {}
    for round_number, round_games in schedule_df.groupby("gameday"):
        turns = assign_round_turns(round_games)
        for game, turn in turns.items():
            lookup[(season, game)] = turn
    return lookup


if __name__ == "__main__":
    lookup = build_turn_lookup()
    schedule_df = load_regular_season_schedule()

    # 14: a single far-off postponement (102-day jump). 30: two separate
    # far-off postponements. 33: the tiebreak case -- two lone games, one
    # 1 day out (real T2), one 7 days out (reschedule) -- same game count,
    # different distance.
    sample_rounds = [14, 30, 33]

    for round_number in sample_rounds:
        round_games = schedule_df[schedule_df["gameday"] == round_number].sort_values(
            "date_parsed"
        )
        print(f"\nRound {round_number}:")
        for _, row in round_games.iterrows():
            turn = lookup[(SEASON, row["game"])]
            print(f"  Game {row['game']:>4}  {row['date']:<13}  ->  {turn}")
