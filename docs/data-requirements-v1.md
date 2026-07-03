# Data Requirements Document — v1

**Project:** EuroLeague Fantasy AI Decision Engine
**Target game:** Official EuroLeague Fantasy Challenge — Classic Mode (General League / global ranking)
**Purpose:** Identify every piece of data the engine needs *before* writing collection code.
**Status:** Draft v1 — rows marked ⚠ need a decision from Moris.

---

## Priority legend
- **Critical** — engine cannot function without it. Build first.
- **High** — needed for good predictions. Build second.
- **Medium** — improves accuracy. Build later.
- **Low / Deferred** — nice-to-have, postponed (matches "Future Extensions" in the plan).

---

## Requirements table

| # | Data Category | Exact Data Needed | Why It Is Needed | Source | Update Frequency | Priority |
|---|---------------|-------------------|------------------|--------|------------------|----------|
| 1 | Fantasy scoring rules | Exact point value per stat action; bench 50% rule; captain 1.5×; coach scoring table | Cannot convert box scores into fantasy points without the exact mapping | Official rules page / rules wiki | Once per season | Critical |
| 2 | Fantasy game constraints | Budget (100 credits), roster shape (4G/4F/2C + coach), trade limits per round, unlimited-trade rounds | These ARE the optimizer's constraints | Official rules page | Once per season (verify per round) | Critical |
| 3 | Player identity & metadata | Player ID, name, current team, fantasy position (G/F/C) | Key that links stats, prices, and roster slots together | EuroLeague site/API | Season start + on transfers | Critical |
| 4 | Player box-score stats (per game) | Pts, reb (off/def), ast, stl, blk, TO, fouls made/drawn, FG/3P/FT made & missed, PIR | The raw inputs the fantasy formula is computed from | EuroLeague stats | After every game | Critical |
| 5 | Fantasy prices | Current credit price per player + price history | Optimization is budget-constrained; value = points per credit | Fantasy game | Every round | Critical |
| 6 | Schedule / fixtures | Round → Turn (game day) structure, date/time, home/away, opponent, double-gameweeks | Drives lineup timing (T1/T2 bench logic) and matchup projections | EuroLeague schedule | Season start + revisions | Critical |
| 7 | Your own team state | Current roster, remaining budget, trades used, coach-change availability | Recommendations are *relative* to your current team; transfers have limits | Fantasy account | Every round | Critical (after initial team) |
| 8 | Minutes played / role | Minutes per game, starter vs bench, rolling minutes trend | Minutes = opportunity; biggest single driver of fantasy output | Box scores (derived) | After every game | High |
| 9 | Player form (derived) | Rolling averages (last 3/5 games) of fantasy pts, minutes, PIR | Recent form predicts near-term output better than season average | Derived from #4 | After every game | High |
| 10 | Injuries / availability | Injury status, expected return, out/doubtful/probable | An unavailable player scores 0 — silently ruins a lineup | News / injury reports ⚠ | Continuous, before each deadline | High |
| 11 | Transfer/trade state | Trades remaining this round, coach-swap option used or not | Optimizer must not recommend illegal transfers | Fantasy account / rules | Every round | High |
| 12 | Historical stats (multi-season) | Same as #4 but for past seasons | Training data for the prediction model | EuroLeague stats archive ⚠ | One-time backfill | High |
| 13 | Opponent / team strength | Team off/def rating, pace, points allowed by position | Matchup difficulty adjusts a player's projection | Team stats (derived) | Rolling, per round | Medium |
| 14 | Suspensions / disciplinary | Suspension status and length | Same effect as injury: player unavailable | News / official ⚠ | Continuous | Medium |
| 15 | Coach / team results | Win/loss, winning margin buckets | Needed to score the coach slot | EuroLeague results | After every game | Medium |
| 16 | Rest / schedule difficulty | Days between games, back-to-backs, travel | Fatigue affects performance | Derived from #6 | Per round | Low / Deferred |
| 17 | Ownership % | % of managers owning each player | Differential strategy vs the field (relevant only near #1) | Fantasy game (if exposed) | Per round | Low / Deferred |
| 18 | News / social signals | Beat-writer news, X/Reddit rotation chatter | Early edge on rotations/injuries | Scraping ⚠ | Continuous | Low / Deferred |

---

## Open decisions (Moris owns these)

**D1 — Data source: API or scrape? (blocks everything)**
Does the EuroLeague site expose a usable stats/schedule API, or must the data be scraped from HTML? This single answer changes the entire Data Collection Engine design. Investigate before writing collection code.

**D2 — How much history to backfill (row 12)?**
How many past seasons of box scores are actually available and consistent? More data = better model, but roster/rule changes reduce older seasons' usefulness. Pick a realistic backfill window.

**D3 — Injuries/news for v1 (rows 10, 14, 18)?**
Structured injury data is the hardest to get and often the biggest edge. Decide for v1: manual entry, scrape, or defer. Recommendation: defer automated news; allow a manual override field so you can zero-out a known-injured player by hand.

---

## Assumptions
- Target is the official Challenge Classic Mode, not Draft mode or a third-party game.
- Prices change over the season (needs confirmation — affects row 5).
- Exact scoring point-values still need to be captured verbatim into row 1.

## Deliberately out of scope for v1
Monte Carlo, Bayesian learning, automated news scraping, ownership-based differential strategy, rest-day modeling. Listed above as Low / Deferred so the plan is complete, but not built yet.
