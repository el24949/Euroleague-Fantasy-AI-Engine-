"""Fetch one game's player box score from euroleague-api and print it."""

from euroleague_api.boxscore_data import BoxScoreData

SEASON = 2023
GAMECODE = 1

boxscore = BoxScoreData()
df = boxscore.get_players_boxscore_stats(SEASON, GAMECODE)

print(df.columns.tolist())
print(df.head())
