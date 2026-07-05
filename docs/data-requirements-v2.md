# Data Requirements Document — v2

**Project:** EuroLeague Fantasy AI Decision Engine
**Target game:** Official EuroLeague Fantasy Challenge — Classic Mode (General League / global ranking)
**Purpose:** Define every piece of data the engine needs *before* writing collection code.
**Status:** v2 — build-ready. Rules verified against the official rulebook; D1 (data source) resolved.

> **What changed from v1:** scoring captured verbatim; captain corrected to ×2; new optimizer constraints added (6-per-team, formations, 6th-man scoring); row 13 raised Medium→High; two new derived-data rows; D1 resolved and D4 added. Structure reorganized so scoring and constraints live in their own sections instead of cramped table cells.

---

## §1 — Scoring reference (verbatim)

The engine converts real box scores into fantasy points here. **This is not PIR** — it is the fantasy game's own formula, plus a team-result bonus PIR does not have.

### Player scoring — every action is worth ±1

| +1 (positive) | −1 (negative) |
|---|---|
| Point scored | Turnover |
| Rebound (offensive or defensive) | Block suffered (blocked shot against) |
| Assist | Foul committed |
| Steal | Missed field goal (2P or 3P) |
| Block (in favour) | Missed free throw |
| Foul drawn (received) | |

**Team Win Bonus:** +10% of the player's round fantasy score **if his team wins the game.**
→ Consequence: a player's score partly depends on his *team's result*, so team win-probability is a genuine scoring input (see row 13).

### Score modifiers
- **Captain:** ×2 on that player's score. *(v1 said ×1.5 — that was wrong.)*
- **Starting five + 6th man:** score at **100%**.
- **Bench (remaining players):** score at **50%**.
- **Coach:** scores at 100% via the coach table below.

### Coach scoring

| Result | Margin | Points |
|---|---|---|
| Win | by 1–10 (or OT) | +10 |
| Win | by 11–20 | +20 |
| Win | by 20+ | +25 |
| Loss | by 1–10 (or OT) | −5 |
| Loss | by 11–20 | −10 |
| Loss | by 20+ | −20 |

> ⚠ **Verify-before-coding note:** the ±1 per-action list and coach margins above are captured from the official rules, but since row 1 is load-bearing, confirm each value once against the live app before hardcoding the scoring function.

---

## §2 — Optimizer constraints (the hard rules the engine must obey)

These are non-negotiable limits. A recommendation that breaks any of these is illegal.

- **Budget:** 100 credits per team.
- **Roster shape:** 10 players (4 Guards / 4 Forwards / 2 Centers) + 1 Head Coach.
- **Max 6 players from the same EuroLeague club** (applies in playoffs too). *(New — not in v1.)*
- **Legal starting-five formations** (Guards-Forwards-Centers): `2-2-1`, `1-2-2`, `2-1-2`, `1-3-1`, `3-1-1`. *(New — v1 treated the lineup as "start your best 10.")*
- **Scoring weights:** starting five + 6th man = 100%; bench = 50%; coach = 100%.
- **Trades:** up to **4 per round**. Trading the coach counts as one of the 4 (so 3 players + coach, or 4 players).
- **Unlimited-trade windows:** after Rounds 5, 11, 17, 24, 28, 34, and throughout Play-in / Playoffs / Final Four.
- **Substitutions:** allowed between turns (T1 / T2) within a round — lineup, captain, and bench can change between turns.

---

## §3 — Requirements table

### Priority legend
- **Critical** — engine cannot function without it. Build first.
- **High** — needed for good predictions. Build second.
- **Medium** — improves accuracy. Build later.
- **Low / Deferred** — nice-to-have, postponed.

| # | Data Category | Exact Data Needed | Why It Is Needed | Source | Update Frequency | Priority |
|---|---------------|-------------------|------------------|--------|------------------|----------|
| 1 | Fantasy scoring rules | Full scoring model (see §1) | Converts box scores into fantasy points | Official rules | Once per season | Critical |
| 2 | Fantasy game constraints | Budget, roster shape, formations, trade limits (see §2) | These ARE the optimizer's constraints | Official rules | Once per season (verify per round) | Critical |
| 3 | Player identity & metadata | Player ID, name, team, fantasy position (G/F/C) | Links stats, prices, and roster slots together | EuroLeague API | Season start + transfers | Critical |
| 4 | Player box-score stats (per game) | Pts, reb (off/def), ast, stl, blk (for/against), TO, fouls (made/drawn), FG/3P/FT made & missed | Raw inputs for the §1 formula — **collect components, not PIR** | EuroLeague API | After every game | Critical |
| 5 | Fantasy prices | Current credit price per player + price history | Optimization is budget-constrained; value = points per credit | Fantasy endpoint | Every round | Critical |
| 6 | Schedule / fixtures | Round → Turn structure, date/time, home/away, opponent, double-gameweeks | Drives lineup timing (T1/T2 bench logic) and matchups | EuroLeague API | Season start + revisions | Critical |
| 7 | Your own team state | Current roster, remaining budget, trades used, coach-swap availability | Recommendations are *relative* to your team; transfers are limited | Fantasy account (login-gated) | Every round | Critical (after initial team) |
| 8 | Minutes played / role | Minutes per game, starter vs bench, rolling minutes trend | Minutes = opportunity; biggest single driver of output | Box scores (derived) | After every game | High |
| 9 | Player form (derived) | Rolling averages (last 3/5 games) of fantasy pts, minutes | Recent form predicts near-term output better than season avg | Derived from #4 | After every game | High |
| 10 | Injuries / availability | Injury status, expected return, out/doubtful/probable | An unavailable player scores 0 — silently ruins a lineup | News / injury reports ⚠ | Continuous, before each deadline | High |
| 11 | Transfer/trade state | Trades remaining this round, coach-swap used or not | Optimizer must not recommend illegal transfers | Fantasy account / rules | Every round | High |
| 12 | Historical stats (multi-season) | Same as #4 for past seasons | Training data for the prediction model | EuroLeague API (back to 2007) | One-time backfill | High |
| 13 | Opponent / team strength | Team off/def rating, pace, **fantasy points allowed by position**, home/away win rate | Matchup difficulty adjusts projections AND drives the +10% win bonus & coach pick | Team stats (derived) | Rolling, per round | **High** ⬆ *(was Medium)* |
| 14 | Projected price change | Expected price delta next round, from (projected score − priced-in expectation) & starting value | Cheap risers grow budget → more points long-term (capital gains) | Derived / learned (see D4) | Per round | **High** *(new)* |
| 15 | Teammate-injury → minutes link | When player X is out, which teammates inherit his minutes/role | Single most-cited edge: injured star → cheap replacement surges | Derived from #4, #8, #10 | Continuous | **High** *(new)* |
| 16 | Suspensions / disciplinary | Suspension status and length | Same effect as injury: player unavailable | News / official ⚠ | Continuous | Medium |
| 17 | Coach / team results | Win/loss, winning-margin buckets | Needed to score the coach slot (see §1) | EuroLeague API | After every game | Medium |
| 18 | Rest / schedule difficulty | Days between games, back-to-backs, travel | Fatigue affects performance | Derived from #6 | Per round | Low / Deferred |
| 19 | Ownership % | % of managers owning each player | Differential strategy vs the field (relevant only near #1) | Fantasy game (if exposed) | Per round | Low / Deferred |
| 20 | News / social signals | Beat-writer news, X/Reddit rotation chatter | Early edge on rotations/injuries | Scraping ⚠ | Continuous | Low / Deferred |

---

## §4 — Edge cases

**Suspended / rested / changed games**
- If a game is postponed and teams are rested, affected players resolve to their **season average score**.
- The **captain does NOT double** on such rested/postponed games.
- A game cancelled *mid-round* keeps the captain's doubling in effect.
- Games rescheduled but still inside the round's window count normally.

---

## §5 — Open decisions

**D1 — Data source: API or scrape? → RESOLVED: hybrid, mostly API.**
- **Stats + schedule (rows 3, 4, 6, 12, 13, 17):** JSON API at `live.euroleague.net/api/`, semi-official and stable, with a mature Python wrapper (`euroleague-api`). Box scores, play-by-play, schedule, and history back to 2007. → Two collectors needed; this is the easy one.
- **Fantasy side (rows 1, 2, 5, 7):** separate system at `fantasychallenge.euroleague.net`. Prices are fetchable via an undocumented `players.json`-style endpoint (find it by inspecting network traffic). **Row 7 (your team state) is login-gated — the hardest single piece.**

**D2 — How much history to backfill (row 12)?** *(still open)*
API has data back to 2007, but rule/roster changes reduce old seasons' usefulness. Pick a realistic window (recommend last 3–5 seasons) during the prediction phase.

**D3 — Injuries/news for v1 (rows 10, 16, 20)?** *(still open)*
Hardest data to get automatically. Recommendation: defer automated news; add a **manual override field** to zero-out a known-injured player by hand.

**D4 — Price model: formula or learned? → learned. *(new)***
The official rules state the game "has no memory" — only the most recent game moves a player's value, based on his score vs expectation and his starting value. The exact formula is **undisclosed**, so row 14 must be **learned empirically** (regression on historical score + starting value → price delta) during the prediction phase, not hardcoded.

---

## §6 — Assumptions & scope

**Assumptions**
- Target is the official Challenge Classic Mode (not Draft mode or a third-party game).
- Prices change over the season — **confirmed** (dynamic, single-game basis).
- "Chips" (bench boost, triple captain, wildcards) mentioned in third-party blogs are **NOT** in the official rules — ignore unless seen in-app.

**Out of scope for now** (listed for completeness, not built yet)
- Monte Carlo / Bayesian modeling, automated news scraping, ownership-based differential strategy, rest-day fatigue modeling.
- **Empirical validation / backtesting** ("which features actually predicted top scores") belongs to the **prediction phase** — it needs the data pipeline flowing first, and rows 9/12/13 already provide for it. Not a requirements-phase task.
