# EuroLeague Fantasy AI Decision Engine

Recommends the best EuroLeague Fantasy Challenge team (Classic Mode) for each
gameweek. The aim is the most expected fantasy points over the season, with
less exposure to players whose output swings a lot from game to game.

The engine is a pipeline. Each stage is a small, separately checked module:

```
data collection ──► SQLite DB ──► features ──► prediction ──► optimization ──► Streamlit UI
(euroleague-api,    (boxscores,   (rolling     (linear       (ILP: roster,     (my team,
 price Excel)        positions,    form,        regression)   lineup, captain,  trades,
                     prices)       minutes,                   risk, turns)      coach)
                                   matchup,
                                   turns)
```

**Headline result.** In a backtest on 2025 rounds 12–34, the engine averaged
**112.4 pts/round**. A naive "pick by season average" optimizer averaged
**106.9**, and a random legal team averaged **62.6**. The engine beat the naive
optimizer in 14 of 23 rounds. Almost all of that edge comes from **avoiding
volatile busts** (the risk penalty), not from better point predictions.

---

## 1. What is being predicted: the fantasy score

The target is the **official EuroLeague Fantasy score**. It is **not** PIR
(`Valuation`) and the engine never uses PIR. Each box-score action is worth ±1
(`scoring/base_score.py`):

| +1 for each                         | −1 for each                               |
|-------------------------------------|-------------------------------------------|
| Point scored                        | Turnover                                  |
| Offensive rebound                   | Shot of yours that gets blocked (`BlocksAgainst`) |
| Defensive rebound                   | Foul committed                            |
| Assist                              | Missed 2-pointer                          |
| Steal                               | Missed 3-pointer                          |
| Block made (`BlocksFavour`)         | Missed free throw                         |
| Foul drawn (`FoulsReceived`)        |                                           |

Misses are not stored in the data. They are worked out as `Attempted − Made`.

**Season rules** (`scoring/season_rules.py`). Anything that changes between
seasons is looked up with `get_rules(season)`. That function raises an error for
an unknown season instead of guessing. Today the only such rule is the captain
multiplier:

| Season | Captain multiplier |
|--------|--------------------|
| 2023   | ×2.0               |
| 2024   | ×1.5               |
| 2025   | ×2.0               |
| 2026   | ×2.0 (checked against the live app at launch) |

---

## 2. Data

All data is stored in `data/euroleague.db` (SQLite).

| Table / source | What it holds | How it gets there |
|---|---|---|
| `boxscores` | One row per player per game, with every counting stat. Seasons 2023 and 2025 (2026 is added as games are played). De-duplicated on `(Season, Gamecode)`. | `data_collection/fetch_season.py` via `euroleague-api`. Future games are skipped by date. |
| `positions` | G / F / C for every player, with 100% coverage for both seasons. | `data_collection/fetch_positions.py`, using `GameStats.get_game_stats` (the leaderboard endpoint only covered about 68%). |
| `prices` | Fantasy price ("Quotation") for each player in each round. | **Manual step:** there is no public API because prices sit behind a login. You download the round's price Excel from the Fantasy site and save it to `data/prices/round_NN/players_stats.xlsx`, or upload it in the app. |
| Schedule | Game dates, used to tell the round's first playing day (T1) from its second (T2). | `euroleague_api.schedule`. Rebuilt in memory each run and not saved. |
| `data/unavailable.csv` | Players you know are out or doubtful. Edited by hand. | `optimizer/availability.py` drops them from the candidate pool. |

**Data checks that found real bugs:**
- The 2025 price files were mislabeled by **+3 rounds**. Each file's real round
  was confirmed by Spearman correlation against box scores, and the result is
  recorded in `data/prices/price_file_provenance.csv`. Don't trust a price
  file's name.
- The price sheet uses different team codes from the box scores (e.g. `RMB`
  instead of `MAD` for Real Madrid). The mapping in `optimizer/coach_ranking.py`
  was built from data by cross-referencing players, not guessed.
- Gamecodes restart at 1 every season. Any join or rolling window keyed on
  `Gamecode` alone mixes up unrelated games from different seasons.

---

## 3. What the model takes into account (features)

Every feature uses **only games before the one being predicted**. Rolling
windows are always `shift(1)` first, so a game never sees its own result.
Every window is grouped by **`(Season, Player_ID)`** or **`(Season, Team)`**,
so it resets at the start of each season.

| Feature | File | Meaning | Why it matters |
|---|---|---|---|
| `rolling_5` | `features/rolling_form.py` | Mean fantasy score over the player's previous 5 games | Current form. This is the strongest signal. |
| `rolling_3` | `features/rolling_form.py` | Mean fantasy score over the previous 3 games | Shorter, faster-reacting form |
| `rolling_5_minutes` | `features/minutes.py` | Mean minutes over the previous 5 games | Role and playing time. Fantasy points need minutes. |
| `rolling_3_minutes` | `features/minutes.py` | Mean minutes over the previous 3 games | Picks up recent role changes (promotion, benching, return from injury) |
| `consistency_5` | `features/rolling_form.py` | Standard deviation of fantasy score over the previous 5 games | Volatility. Used by the model **and** as the optimizer's risk penalty. |
| `distance_from_norm` | `features/rolling_form.py` | How far the player's *last* game was above or below their season-to-date average at that point | Momentum, or whether the last game was a fluke |
| `opp_rolling_5_conceded` | `features/matchup.py` | Total fantasy points the **opponent** gave up per game over their previous 5 games | Matchup. A weak defense gives up more fantasy points. |

`DNP` counts as 0 minutes. A player needs a value for **every** feature to get a
prediction, which in practice means at least two prior games this season.
Season debuts are dropped. The engine can't see them yet.

**Turn tags** (`features/turn_tags.py`) are not a model feature. They feed the
optimizer. Each round is played over two days. The two dates with the most
games become T1 (earlier) and T2 (later). Any other date is marked
`RESCHEDULED`, for postponed or broadcast-shifted games, instead of being forced
into a turn. Ties are broken by distance from the main date (round 33 of 2025
is the test case).

---

## 4. Prediction model

**Model:** `sklearn.linear_model.LinearRegression` on the 7 features above.
Target: `base_score` (fantasy points for that game).

**Fitted coefficients** (2025, trained on Gamecode < 304; `python -m prediction.model_v3`):

```
predicted_score = −1.678
                + 0.397 × rolling_5
                + 0.069 × rolling_3
                + 0.167 × rolling_5_minutes
                + 0.060 × rolling_3_minutes
                − 0.033 × consistency_5
                + 0.023 × opp_rolling_5_conceded
                − 0.025 × distance_from_norm
```

What the coefficients say:
- **Recent form and minutes do almost all the work.** A player at 15 pts over
  their last 5 games and 28 minutes per game is predicted at roughly 13–14 pts.
  The model pulls everyone toward the mean (regression to the mean), which is
  correct behaviour for noisy single-game scores.
- **Matchup is a small adjustment.** Facing a defense that gives up 20 more
  fantasy points per game than average adds about +0.5 pts.
- **`distance_from_norm` is slightly negative.** One unusually big game tends
  to be followed by a partial drop back toward normal.
- **`consistency_5` is nearly zero inside the model.** Volatility doesn't change
  the *expected* score much, but it changes the *risk* a lot. That's why the
  optimizer penalizes it separately (section 5).

**Accuracy:** test **MAE 5.506** fantasy points per player-game. This uses a
2025-only chronological split: the first ~80% of games train, the last ~20%
test, and the split is never random. An older figure of 5.22 came from a leaky
split that mixed seasons and can't be compared with this one.

**In production** (`backtest/simulate_round.py → build_predictions`, used by
`pipeline/update_round.py` and the app), the model is **retrained every round**
on all games of the season *before* the target round. The target round's stats
are blanked (set to 0 / `DNP`) before features are computed, and assertions
confirm nothing leaked.

**Things that were tried and rejected, with evidence:**
- Random Forest: 9 configurations, regularized. It never beat the linear model.
  The features limit accuracy, not the model.
- Injury/availability data: a ceiling test showed about 0 benefit, because the
  minutes features already pick up most availability effects.
- Per-position matchup: too weak a signal.

---

## 5. Optimizer: building the team

`optimizer/pick_lineup_ilp.py → solve_lineup()` is a single **mixed-integer
linear program** (PuLP + CBC). It chooses the roster *and* the lineup together.

### What each player is worth to the optimizer

```
objective_score = predicted_score − RISK_LAMBDA × consistency_5        (RISK_LAMBDA = 0.25)
```

This is the risk-aware part. Two players predicted at 14 pts are not equal if
one scores 13–15 every week and the other swings between 2 and 30. The steady
player wins. λ = 0.25 came from a backtest sweep over λ ∈ {0, 0.1, 0.25, 0.5, 1.0}:

| λ | Avg actual pts/round (2025 R12–34) | Rounds beating naive |
|---|---|---|
| 0.00 | 107.5 | 10 / 23 |
| **0.25** | **112.4** | **14 / 23** |

### Hard constraints (the official game rules)

| Rule | Constraint |
|---|---|
| Roster | Exactly 10 players: **4 Guards, 4 Forwards, 2 Centers** |
| Coach | 1 coach. The coach's price is subtracted from the budget first. |
| Budget | Total player price ≤ 100 − coach price (editable in the app) |
| Club limit | At most **6 players** from any one club |
| Starting five | Exactly 5 starters, in one legal formation: `2-2-1`, `1-2-2`, `2-1-2`, `1-3-1`, `3-1-1` (G-F-C) |
| 6th man | Exactly 1, any position |
| Captain | Exactly 1, chosen from the starters (assumption: the rulebook doesn't say whether a bench player can be captain) |

### Objective (what is maximized)

```
  1.0 × Σ objective_score  (5 starters + 6th man)
+ 0.5 × Σ objective_score  (4 bench players: they count at 50%)
+ (captain_multiplier − 1) × objective_score(captain)
+ 0.5 × (number of position groups G/F/C that have both a T1 and a T2 player)
```

The last term is the **turn-coverage bonus**. If each position group has
players on both days, you can still substitute after seeing T1 results. The
bonus is worth at most 1.5 points, so it only breaks ties between near-equal
rosters. *It is reasoned, not backtested.* Its real payoff comes from mid-round
substitutions, and the static backtest doesn't simulate those.

### Coach

`optimizer/coach_ranking.py` ranks teams by a **leakage-free rolling 5-game real
point margin** (points scored − points conceded). Home games win ties. It shows
the top 5 coaches and **you choose**. The default is #1. This is a rough measure
of team strength, not a fantasy-points model.

### Trade suggestions

`optimizer/suggest_trades.py` starts from **your current squad** and solves a
second ILP. It finds up to N one-for-one swaps that maximize total
`objective_score`, under the same position, budget and club constraints.
Players the engine can't see (no prediction yet) are **pinned**. They're never
forced out and never suggested in, because the engine has no basis for judging
them. The coach is never changed by this tool.

---

## 6. Validation: the backtest

`backtest/simulate_many.py` replays every 2025 round from 12 to 34. For each
round:
1. Features and the model are rebuilt from games **before** that round only.
2. Three legal teams are picked under identical constraints: **engine**
   (risk-aware predictions), **naive** (season-to-date average) and **random**.
3. Each team is scored on what actually happened in that round.

| Strategy | Avg pts / round |
|---|---|
| Engine (λ = 0.25) | **112.4** |
| Naive season average | 106.9 |
| Random legal team | 62.6 |

The prediction model alone isn't much better than a season average. The gain
comes from combining it with the **variance penalty**, which keeps
high-average, high-variance players out of the team.

---

## 7. Known limitations

- **Cold start in 2026.** About 36 players, including NBA returnees and
  rookies, have **no EuroLeague history**. The engine can't see them until they
  have about two 2026 games. Pick early rounds partly by hand.
- **One season of tuning.** λ = 0.25 was tuned on 2025 only and should be
  re-checked as 2026 data comes in.
- **Turn bonus untested.** See section 5.
- **Coach choice is crude.** It uses team margin, not coach fantasy points.
- **New club codes.** BJK (Beşiktaş) isn't in the 2025-derived team-code map
  yet. Refresh the map once 2026 data exists.
- **Captain must be a starter.** This hasn't been checked against the live app.

---

## 8. Running it

```bash
pip install -r requirements.txt     # note: the file is UTF-16 encoded

# whole engine, one command (refresh data → prices → predict → optimize → print team)
python -m pipeline.update_round

# web UI: My Team, trade suggestions, price upload, data refresh, coach choice
streamlit run app.py
```

Always run scripts as modules (`python -m package.module`) so imports resolve.
Each round:
1. Click **Update game data** in the app, or run the pipeline.
2. Download the round's price Excel from the Fantasy site and upload it under
   **Manual → Upload round prices**.
3. Add any known absences to `data/unavailable.csv`.
4. Open **My Team → Suggest trades**, review the swaps, and apply them.

### Project layout

```
scoring/          fantasy score formula + per-season rules
data_collection/  box scores, positions, prices (API + manual Excel)
features/         rolling form, minutes, matchup defense, T1/T2 turn tags
prediction/       linear models (model_v3 = current), baselines, RF experiment
optimizer/        ILP lineup solver, coach ranking, my team, trade suggestions
backtest/         round-by-round historical simulation
pipeline/         update_round.py: the end-to-end per-round command
app.py            Streamlit frontend
```
