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
Phase 4: OPTIMIZER (in progress). Collection, scoring, features, prediction all done.

DATA: 2023 regular season in data/euroleague.db, `boxscores` (306 games, 8,508
rows, 29 cols). Collector re-run-safe (de-dup on Season+Gamecode). Source:
euroleague-api. NOT yet collected: fantasy prices + team state (rows 5/7,
login-gated endpoint) and player positions (row 3) — both required to go live.

SCORING (scoring/): base_score.py (±1 per action, hand-verified). season_rules.py
— get_rules(season), fails loudly on unknown season. Captain ×1.5 (2024), ×2 (2025).
NOT PIR — never use the Valuation column. Misses are derived (Attempted − Made).

FEATURES (features/): all leakage-free (rolling windows shifted to prior games only).
minutes.py (rolling_3/5_minutes), rolling_form.py (rolling_3/5), matchup.py
(rolling_5 team defensive rating, opponent-verified), plus consistency_5 and
distance_from_norm.

PREDICTION (prediction/): naive baseline MAE 5.40 → linear v3 (7 features): 2025-only chronological split, **MAE 5.506** ← current baseline.
(Old 2023 single-season number was 5.22 but measured on a now-fixed blended/leaky
split — not comparable.) DB now holds 2023+2025; all rolling features keyed on
(Season, Player_ID/Team) to prevent cross-season blending.
 Random Forest tested + regularized across 9 configs:
never beat linear. FINDING: model is not the bottleneck, features are. Single-season
box-score signal is ~tapped out; each new feature buys only ~0.03.

OPTIMIZER (optimizer/): ILP via PuLP/CBC. Done: 4G/4F/2C roster shape, budget ≤100
(binding, spends 100.0), max 6 per club (tested with cap=2). Positions and prices are
DETERMINISTIC PLACEHOLDERS — flagged in code, must be replaced by real data.
IN PROGRESS: Slice A — starting five + 6th man at 100%, bench at 50%, legal formations
(2-2-1/1-2-2/2-1-2/1-3-1/3-1-1) via one-active binary per formation.
NEXT: Slice B — captain ×2 via get_rules(season).

DEFERRED (with reasons): per-position matchup (overall version weak, coef 0.024; needs
real positions + established rosters). Multi-season collection (2021/2022/2024) — the
big lever: more real history helps every feature, fixes cold-start, enables career
features. Domestic-league games (expensive, low payoff). Player-vs-team history
(statistically thin, ~2 games/season — likely noise). Ownership % → optimizer objective,
not a prediction feature.

GOTCHAS: run as modules (python -m prediction.model). requirements.txt is UTF-16.
Reuse existing scoring/feature functions — never recompute or reimplement.
Live target is 2026-27 (unplayed); 2023 is the validation ground.

METHOD THAT WORKS: smallest verifiable slice → hand-check the numbers myself →
commit → next. Every feature is judged against test MAE on an identical time-based
split. No hand-tuned weights; the model decides.
KEY FINDING: risk-aware optimization (objective = predicted − λ·consistency_5) beats naive. λ≈0.25 → 112.4 vs naive 106.9, 14/23 rounds. The edge is avoiding volatile busts, NOT better prediction. λ is data-specific (peak for 2025 rounds 12–34); re-tune with more seasons






