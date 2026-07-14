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
Phase 3: prediction (in progress). Data collection + scoring + core features done.

DATA (data_collection/): full 2023 regular season in data/euroleague.db,
table `boxscores` (306 games, 8,508 rows, 29 cols). Collector is re-run-safe
(de-dup guard on Season+Gamecode). Source: euroleague-api (fantasy price
endpoint still to come).

SCORING (scoring/): base_score.py — ±1-per-action fantasy score, hand-verified.
season_rules.py — season-aware multipliers via get_rules(season), fails loudly
on unknown seasons. Captain ×1.5 (2024), ×2 (2025). NOT PIR — do not use the
Valuation column. Misses are derived (Attempted − Made).

FEATURES (features/): all leakage-free (rolling windows shifted to prior games
only). minutes.py — rolling_3/5_minutes. rolling_form.py — rolling_3/5 fantasy
form. matchup.py — rolling_5 team defensive rating (fantasy pts conceded to
opponents), opponent-verified.

PREDICTION (prediction/): naive_predict.py — rolling_5 baseline (MAE 5.40).
evaluate.py — MAE harness. model.py — linear regression (form+minutes),
time-based train/test split, MAE 5.28. Currently adding opponent matchup as
a 3rd feature.

CONVENTIONS THAT KEEP BITING: run feature/model files as modules
(python -m prediction.model) due to package imports. requirements.txt is UTF-16.
Reuse existing feature/scoring functions — never recompute or reimplement.

Next: finish matchup feature eval, then more features / better model, then the
OPTIMIZER (pick best legal team: 100 credits, max 6/club, formations, captain)
and season backtest. Live target season is 2026-27 (unplayed); 2023 is the
validation ground.
Per-position matchup deferred: overall version weak (coef 0.024), but per-position defense (e.g. weak vs centers) likely matters — needs player positions (row 3) + established rosters. Revisit for live season.

