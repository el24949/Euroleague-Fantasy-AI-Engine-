"""Slice 3: Streamlit display of the recommended team, with a round selector.

ONE input widget only (round selectbox, values restricted to a known-safe range
[30-36]) -- no team entry, no budget, no trades, no persistence, no DB write.
Everything downstream of load_recommendation()'s returned dict is unchanged from
slice 2: still exactly the plain dict get_recommended_team() (pipeline/update_round.py)
returns; nothing here recomputes a feature, a price, or the lineup.

WHY load_recommendation() IS THE ONLY CACHED FUNCTION, AND WHY IT'S THE ONLY
THING CALLED AT MODULE LEVEL: Streamlit reruns this entire script top-to-bottom
on every user interaction (a widget change, a browser refresh). Building Step 6's
inputs is heavy -- run_prediction_step() alone retrains the linear model on every
prior Season 2025 game -- so it must not run on every rerun. @st.cache_data on
this one function makes Streamlit run its body once and replay the cached return
value on subsequent reruns. get_recommended_team() itself is NOT decorated: it's
a pure function of the (already-cached) inputs this file passes it, so caching
it separately would add nothing. No other work happens at import time -- the
functions below the imports are all `def`s; the only top-level *calls* are
load_recommendation() and the st.* render calls that follow it.

SAME PATTERN pipeline/update_round.py's __main__ uses: Season 2025, a past round
standing in for "the upcoming round" (see that module's own "TESTING ON A PAST
ROUND" docstring section for why -- Season 2025 is fully played). round_no here
is user-selected instead of the module's fixed TARGET_ROUND, restricted to
ROUND_OPTIONS so a stray value can't hit an unpriced/uncollected round.

NO DB WRITE: load_prices_for_round() is called directly here, never store_prices()
(the only function in that module that writes to the prices table) -- see
data_collection/load_prices_history.py.
"""

import streamlit as st

from data_collection.load_prices_history import load_prices_for_round
from pipeline.update_round import (
    SEASON,
    get_recommended_team,
    get_round_gamecodes,
    run_prediction_step,
)

st.set_page_config(page_title="EuroLeague Fantasy -- Recommended Team", layout="wide")

ROUND_OPTIONS = [30, 31, 32, 33, 34, 35, 36]


@st.cache_data
def load_recommendation(round_no):
    """Step 6's inputs, built exactly as update_round.py's __main__ does, then
    get_recommended_team()'s plain dict. Cached per round_no argument -- Streamlit
    reruns this script top-to-bottom on every widget interaction, so round_no MUST
    arrive as a function argument (not read from st.session_state inside this
    function) for @st.cache_data to key its cache on it and skip recompute when
    the user switches back to an already-seen round."""
    target_gamecodes = get_round_gamecodes(SEASON, round_no)
    prices_df, _coverage = load_prices_for_round(round_no, season=SEASON)
    round_features = run_prediction_step(target_gamecodes)
    return get_recommended_team(round_features, target_gamecodes, prices_df, season=SEASON)


st.caption(
    "This displays a 2025 VALIDATION team, not a live 2026 pick -- 2026 box scores "
    "are still empty this early in the season, so the engine has no game logs to "
    "predict from yet. This screen is plumbing/proof that the pipeline renders "
    "correctly, not a recommendation to act on."
)

st.caption("Season: 2025")
round_no = st.selectbox("Round", ROUND_OPTIONS, index=ROUND_OPTIONS.index(32))

team = load_recommendation(round_no)

st.title("Recommended Team")

header_cols = st.columns(4)
header_cols[0].metric("Formation", team["formation"])
header_cols[1].metric("Solver status", team["solver_status"])
header_cols[2].metric("Captain multiplier", f"x{team['captain_multiplier']}")
header_cols[3].metric(
    "Total squad cost",
    f"{team['total_squad_cost']:.1f} / {team['budget']}",
)

coach = team["chosen_coach"]
budget_cols = st.columns(3)
budget_cols[0].metric(
    "Chosen coach",
    f"{coach['name']} ({coach['team']})",
    help=f"Price {coach['price']:.1f}",
)
budget_cols[1].metric("Player budget", f"{team['player_budget']:.1f}")
budget_cols[2].metric("Season", team["season_resolved"])

st.subheader("Full 11-slot team")
full_team_columns = [
    "Player",
    "role",
    "position",
    "turn",
    "price",
    "predicted_score",
    "consistency_5",
    "objective_score",
]
st.dataframe(
    [{col: row[col] for col in full_team_columns} for row in team["full_team"]],
    width="stretch",
)

st.subheader("Coach shortlist")
st.dataframe(team["coach_shortlist"], width="stretch")

st.subheader("Turn coverage by position")
st.dataframe(team["turn_coverage_by_position"], width="stretch")

st.subheader("Players the engine is blind to")
st.caption(
    "Excluded before the optimizer ever saw them -- manual availability overrides "
    "and candidates with no matching price this round, not engine picks."
)
excluded_cols = st.columns(2)
excluded_unavailable = team["excluded_unavailable"]
excluded_no_price = team["excluded_no_price"]
excluded_cols[0].markdown(f"**Unavailable (manual override): {excluded_unavailable['count']}**")
excluded_cols[0].write(excluded_unavailable["names"] or "None")
excluded_cols[1].markdown(f"**No matching price this round: {excluded_no_price['count']}**")
excluded_cols[1].write(excluded_no_price["names"] or "None")
