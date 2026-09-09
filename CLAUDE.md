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
## Status
Phase 6: LIVE + FRONTEND (in progress). Engine functionally COMPLETE and go-live ready.
2026-27 season is LIVE.

RUN: `python -m pipeline.update_round` = the whole engine as one command (refresh
box scores + positions + turns -> load round prices -> leakage-safe predict ->
risk+turn-aware optimize -> print 11-slot team). Print-only; does NOT persist the
recommendation (writes happen inside the collectors it calls).

DATA: 2023 + 2025 seasons in data/euroleague.db (`boxscores`, de-dup on Season+Gamecode).
Positions table (both seasons, 100% coverage via GameStats.get_game_stats, NOT the
leaderboard endpoint). Prices: verified per-round 2025 history rounds 13-36 in
data/prices/round_NN/ (Spearman-verified vs box scores, provenance manifest in
data/prices/; filenames were mislabeled by +3 rounds - never trust price-file names,
verify by correlation). 2026 round_01 price file ingested.

SCORING (scoring/): base_score.py (±1/action, hand-verified). season_rules.py -
get_rules(season), fails loud on unknown. Captain x1.5(2024), x2(2025), x2(2026).
NOT PIR - never use Valuation. Misses derived (Attempted-Made).

FEATURES (features/): all leakage-free AND cross-season-safe (rolling/groupby keyed
on (Season, Player_ID/Team), never Gamecode alone - gamecodes restart each season).
minutes, rolling_form (rolling_3/5 + consistency_5 + distance_from_norm), matchup
(team def rating, opponent-verified), turn_tags (T1/T2 from schedule dates,
reschedule-safe).

PREDICTION (prediction/): linear model. 2025-only chronological split baseline
**MAE 5.506** (old 5.22 was a leaky/blended split, not comparable). RF tested +
regularized across 9 configs, NEVER beat linear. Features are the bottleneck, not
the model.

OPTIMIZER (optimizer/): ILP (PuLP/CBC), COMPLETE. Full legal 11-slot team (10 players
4G/4F/2C + coach), budget<=100, max 6/club, formations, 6th man, captain, coach budget
deduction. Objective = predicted - RISK_LAMBDA(0.25)*consistency_5 + turn-coverage
bonus. Coach: crude margin shortlist (rank_coaches), human picks from top 5.
solve_lineup() is the reusable core. Positions/prices now REAL (not placeholders).

**KEY FINDING: risk-aware optimization BEATS naive baseline +5.5 pts/round (14/23
rounds). The edge is avoiding volatile busts, NOT better prediction. This is the
headline result.** Backtest: engine 112.4 vs naive 106.9 vs random 62.6 (rounds 12-34,
lambda=0.25). Turn-coverage bonus is reasoned-not-backtested (payoff is via T1/T2
substitution the static optimizer doesn't simulate).

TEAM STATE (optimizer/my_team.py, row 7): loads MY roster from data/my_team.csv,
resolves names against the LIVE price file (2026 box scores empty at launch), recovers
'-' placeholder IDs from 2025 where possible. CSV-based, deferred in favor of frontend.

CURRENTLY BUILDING: Streamlit frontend. Slice 0 = refactor update_round to RETURN the
team object (not just print). Slice 1 = read-only UI that renders it. Then interactive
input (my team, budget, trades, coach choice). GOTCHA: Streamlit reruns on every
interaction - modules with import-time network/DB side effects (fetch_positions) must
be behind cached functions, not triggered by import.

2026 LAUNCH REALITY: ~36 league players (incl. Valanciunas) are NBA/rookie signings
with ZERO EuroLeague history - engine is BLIND to them until they build 2026 game logs.
So early rounds are picked MANUALLY; engine takes over as form accumulates. Cold-start
is worse than usual this season for this reason. BJK (Besiktas) is a 2026 club missing
from the 2025-derived team-code map - refresh the map once 2026 data exists.

DEFERRED (decided with evidence, do NOT re-litigate): injury/availability data
(ceiling-tested, ~0 benefit - engine infers from minutes); non-linear models (never
beat linear); per-position matchup (weak, needs real positions + established rosters);
multi-season collection (2021/2022/2024); domestic-league stats (separate source, not
automatic); exact 2025 backtest on real per-round prices (data now exists, not yet run).

GOTCHAS: run as modules (python -m ...). requirements.txt is UTF-16. Reuse existing
scoring/feature/optimizer functions - never recompute. Some collectors run work at
IMPORT time (use importlib or refactor to functions). 2023 = validation ground,
2025 = price/backtest data, 2026 = live target.

METHOD THAT WORKS: smallest verifiable slice -> hand-check numbers myself -> commit ->
next. Every feature judged vs test MAE on identical time-based split. No hand-tuned
weights; the model decides. Verify every join before trusting it (this caught: the
whitespace bug, RMB/MAD team codes, +3 price-file mislabeling, the 68% position gap).