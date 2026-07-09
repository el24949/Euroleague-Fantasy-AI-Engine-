\# EuroLeague Fantasy AI Decision Engine



\## Goal

Recommend the optimal EuroLeague Fantasy team each gameweek to maximize

expected fantasy points over the season. Target game: official EuroLeague

Fantasy Challenge, Classic Mode (global General League ranking).



\## Owner

Moris — ECE student at NTUA, learning Claude Code. I want to UNDERSTAND my

own project, not just run generated code.



\## Architecture (planned)

Data collection -> database -> feature engineering -> prediction ->

optimization -> explanation -> dashboard.



\## Rules for Claude Code

\- Build in small vertical slices. No giant multi-file dumps.

\- Explain each new file's purpose before and after writing it.



\## Status

Phase 1: data collection. Data Requirements Doc v2 done — rules verified,
D1 resolved (hybrid: euroleague-api for stats, fantasy endpoint for prices later).

Working so far (data_collection/):
- fetch_one_boxscore.py — pull one game's box score
- store_boxscore.py — persist a game to SQLite (data/euroleague.db)
- fetch_one_round.py — loop a full round, append to the table
- fetch_season.py — loop a whole season with a de-dup guard (skips games
  already stored by Season+Gamecode, so it's safe to re-run)

Data lives in data/euroleague.db, table `boxscores` (29 columns, verified
to contain every input the scoring formula needs).

Next: feature engineering (minutes, rolling form, matchup strength), then a
season-aware scoring function — scoring rules differ per season (captain
×1.5 in 2024, ×2 in 2025), so multipliers must be looked up by season, not
hardcoded.

