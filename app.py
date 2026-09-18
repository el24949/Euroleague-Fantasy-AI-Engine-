"""Slice 2: read-only Streamlit display of the recommended team.

DISPLAY ONLY -- no input widgets, no persistence, no team entry, no DB write.
Renders exactly the plain dict get_recommended_team() (pipeline/update_round.py)
already returns; nothing here recomputes a feature, a price, or the lineup.

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

SAME ROUND pipeline/update_round.py's __main__ targets: Season 2025, TARGET_ROUND
(see that module's own "TESTING ON A PAST ROUND" docstring section for why --
Season 2025 is fully played, so this stands in for "the upcoming round").

NO DB WRITE: load_prices_for_round() is called directly here, never store_prices()
(the only function in that module that writes to the prices table) -- see
data_collection/load_prices_history.py.
"""

import streamlit as st

from data_collection.load_prices_history import load_prices_for_round
from pipeline.update_round import (
    SEASON,
    TARGET_ROUND,
    get_recommended_team,
    get_round_gamecodes,
    run_prediction_step,
)

st.set_page_config(page_title="EuroLeague Fantasy -- Recommended Team", layout="wide")


@st.cache_data
def load_recommendation():
    """Step 6's inputs, built exactly as update_round.py's __main__ does, then
    get_recommended_team()'s plain dict. Cached so this only runs once per
    Streamlit session, not on every rerun."""
    target_gamecodes = get_round_gamecodes(SEASON, TARGET_ROUND)
    prices_df, _coverage = load_prices_for_round(TARGET_ROUND, season=SEASON)
    round_features = run_prediction_step(target_gamecodes)
    return get_recommended_team(round_features, target_gamecodes, prices_df, season=SEASON)


st.caption(
    "This displays a 2025 VALIDATION team, not a live 2026 pick -- 2026 box scores "
    "are still empty this early in the season, so the engine has no game logs to "
    "predict from yet. This screen is plumbing/proof that the pipeline renders "
    "correctly, not a recommendation to act on."
)

team = load_recommendation()

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
