"""Slice 4: Streamlit display of the recommended team, plus a display-only squad
picker (round selector from slice 3 unchanged).

The squad picker section is a multiselect that echoes back what's picked --
count, total price, position breakdown. NO validation (no roster-shape check,
no budget cap, no legality check) and NO persistence (no my_team.csv write, no
DB write, no optimizer call). It reads the live price table via
optimizer.my_team.load_prices() and does nothing else with it.

ROUND SELECTOR (slice 3): ONE input widget, values restricted to a known-safe
range [30-36] -- no team entry, no budget, no trades. Everything downstream of
load_recommendation()'s returned dict is unchanged from slice 2: still exactly
the plain dict get_recommended_team() (pipeline/update_round.py) returns;
nothing here recomputes a feature, a price, or the lineup.

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
from features.turn_tags import build_turn_lookup
from my_team_writer import read_header_comments, render_csv, save_my_team
from optimizer.my_team import POSITION_NAME_TO_CODE, load_prices
from optimizer.pick_lineup_ilp import RISK_LAMBDA, solve_lineup
from optimizer.turn_join import attach_turns
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
    round_features = load_round_features(round_no)
    return get_recommended_team(round_features, target_gamecodes, prices_df, season=SEASON)


@st.cache_data
def load_round_features(round_no):
    """run_prediction_step() behind its own cache so the squad-roles section
    (slice 6) and load_recommendation share ONE model fit per round."""
    return run_prediction_step(get_round_gamecodes(SEASON, round_no))


def assign_squad_roles(round_features, selected_df):
    """Slice 6: run the existing solve_lineup() with the roster FIXED to the
    selected squad. Returns (result_df, error_message); exactly one is None.

    The candidate pool is restricted to the selected Player_IDs AND those rows
    are forced onto the roster (fixed_roster_idx), so only roles + formation are
    optimized. Formation/weights/risk objective come from pick_lineup_ilp
    unchanged. Captain is NOT decided by the ILP (captain_multiplier=1.0 zeroes
    its bonus and its captain pick is ignored): it is a post-step over T1
    starters only. Prices are the live price table's, same as the legality check."""
    selected_ids = set(selected_df["Player_ID"].str.strip())
    pool = round_features[round_features["Player_ID"].str.strip().isin(selected_ids)].copy()
    pool["Player_ID"] = pool["Player_ID"].str.strip()

    missing = selected_df[~selected_df["Player_ID"].str.strip().isin(set(pool["Player_ID"]))]
    if len(missing):
        names = [f"{r.Surname.upper()}, {r.Name}" for r in missing.itertuples()]
        return None, (
            "No prediction available this round for: " + "; ".join(names)
            + " (did not play, or not enough prior history)."
        )
    if pool["Player_ID"].duplicated().any():
        return None, "Duplicate Player_ID rows in this round's features."

    # Position + price both come from the price sheet (the source the slice-5
    # legality check used) -- NOT attach_positions(): box-score positions can
    # disagree with the price sheet for a player, and the squad's legal 4G/4F/2C
    # shape is defined by the price sheet.
    sheet = selected_df.assign(Player_ID=selected_df["Player_ID"].str.strip()).set_index("Player_ID")
    pool["position"] = pool["Player_ID"].map(sheet["Position"].map(POSITION_NAME_TO_CODE))
    pool["price"] = pool["Player_ID"].map(sheet["Quotation"])
    pool = attach_turns(pool, build_turn_lookup(SEASON))
    pool["objective_score"] = pool["predicted_score"] - RISK_LAMBDA * pool["consistency_5"]

    pos_counts = pool["position"].value_counts().to_dict()
    if any(pos_counts.get(p, 0) != n for p, n in REQUIRED_POSITION_COUNTS.items()):
        return None, f"Squad positions don't form 4G/4F/2C: {pos_counts}"

    result = solve_lineup(
        pool,
        "objective_score",
        1.0,  # captain handled below as a post-step, not by the ILP
        budget=BUDGET,
        fixed_roster_idx=pool.index,
    )
    if result["status"] != "Optimal":
        return None, f"Solver status: {result['status']}"

    role_by_idx = {i: "Starter" for i in result["starters_idx"]}
    role_by_idx.update({i: "6th Man" for i in result["sixth_idx"]})
    role_by_idx.update({i: "Bench" for i in result["bench_idx"]})
    pool["role"] = pool.index.map(role_by_idx)

    t1_starters = pool[(pool["role"] == "Starter") & (pool["turn"] == "T1")]
    if t1_starters.empty:
        return None, "No T1 starter to captain -- none of the starters plays in turn 1."
    captain_idx = t1_starters["predicted_score"].idxmax()
    pool["captain"] = pool.index == captain_idx
    pool.attrs["formation"] = result["formation"]
    return pool, None


@st.cache_data
def load_player_table():
    """optimizer.my_team.load_prices() behind a cached function, same pattern as
    load_recommendation -- it parses an Excel file, so it shouldn't run on every
    rerun. load_prices() returns (df, meta_dict) of excluded rows (head coaches,
    missing position, placeholder ID); this slice only needs the usable df, so
    meta_dict is discarded here and never returned."""
    df, _meta = load_prices()
    return df


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

st.divider()
st.header("My Squad (display-only preview)")
st.caption(
    "Pick players below -- this echoes back what you select and reports whether it's "
    "a legal squad. It never blocks a pick and nothing is saved."
)

players_df = load_player_table()
player_names = {
    row.Player_ID: f"{row.Surname.upper()}, {row.Name}"
    for row in players_df.itertuples()
}

selected_ids = st.multiselect(
    "Pick your squad",
    options=players_df["Player_ID"].tolist(),
    format_func=lambda player_id: player_names[player_id],
)

selected_df = players_df[players_df["Player_ID"].isin(selected_ids)]

# Coach picker lives here (not in the save section) because the coach's price is
# part of the 100 budget, so it must be known before the legality check runs.
shortlist = team["coach_shortlist"]
coach_choice = st.selectbox(
    "Coach (from the round's top-5 shortlist above)",
    options=range(len(shortlist)),
    index=None,
    placeholder="Pick a coach (its price counts toward the 100 budget)",
    format_func=lambda i: (
        f"{shortlist[i]['coach']} ({shortlist[i]['Team']}) -- "
        f"price {shortlist[i]['coach_price']:.1f}"
    ),
)
coach_row = shortlist[coach_choice] if coach_choice is not None else None


squad_cols = st.columns(2)
squad_cols[0].metric("Players selected", len(selected_df))
squad_cols[1].metric("Total price", f"{selected_df['Quotation'].sum():.1f}")

st.subheader("Position breakdown")
st.dataframe(
    selected_df["Position"].value_counts().rename_axis("Position").reset_index(name="Count"),
    width="stretch",
)

st.subheader("Roster legality check")
st.caption(
    "Display/feedback only -- reports against the 4 hard constraints below, never "
    "blocks or removes a pick. A partial squad (<10) is a normal in-progress state."
)

REQUIRED_POSITION_COUNTS = {"G": 4, "F": 4, "C": 2}
POSITION_LABELS = {"G": "Guards", "F": "Forwards", "C": "Centers"}
SQUAD_SIZE = 10
MAX_PER_CLUB = 6
BUDGET = 100

failures = []

squad_count = len(selected_df)
if squad_count == SQUAD_SIZE:
    st.write(f"Players: {squad_count}/{SQUAD_SIZE} OK")
else:
    st.write(f"Players: {squad_count}/{SQUAD_SIZE} - need {SQUAD_SIZE}")
    failures.append(f"Players {squad_count}/{SQUAD_SIZE}")

mapped_positions = selected_df["Position"].map(POSITION_NAME_TO_CODE)
for code, required in REQUIRED_POSITION_COUNTS.items():
    actual = int((mapped_positions == code).sum())
    label = POSITION_LABELS[code]
    if actual == required:
        st.write(f"{label}: {actual}/{required} OK")
    else:
        note = "too many" if actual > required else "too few"
        st.write(f"{label}: {actual}/{required} - {note}")
        failures.append(f"{label} {actual}/{required}")

club_counts = selected_df["Team"].value_counts()
clubs_over_limit = club_counts[club_counts > MAX_PER_CLUB]
if clubs_over_limit.empty:
    st.write(f"Club limit: OK (max {MAX_PER_CLUB} per club)")
else:
    for club, count in clubs_over_limit.items():
        st.write(f"{club}: {count}/{MAX_PER_CLUB} - too many")
        failures.append(f"{club} {count}/{MAX_PER_CLUB}")

players_cost = float(selected_df["Quotation"].sum())
coach_cost = float(coach_row["coach_price"]) if coach_row is not None else 0.0
total_cost = players_cost + coach_cost
breakdown = f"Cost: {players_cost:.1f} players + {coach_cost:.1f} coach = {total_cost:.1f}/{BUDGET}"
if total_cost <= BUDGET:
    st.write(f"{breakdown} OK")
else:
    st.write(f"{breakdown} - over budget")
    failures.append(f"Cost {players_cost:.1f} players + {coach_cost:.1f} coach = {total_cost:.1f}/{BUDGET}")

if not failures:
    st.success("Legal squad")
else:
    st.error("Not a legal squad -- " + "; ".join(failures))

st.divider()
st.header("Squad roles (display-only)")
st.caption(
    f"Roster fixed to your 10 players; the existing lineup optimizer assigns "
    f"Starter / 6th Man / Bench + formation (risk objective, lambda={RISK_LAMBDA}). "
    "Captain = highest predicted_score among T1 starters (post-step). Round "
    f"{round_no}, Season {SEASON}. Nothing is saved."
)

if failures:
    st.info("Complete a legal squad above to assign roles.")
    st.button("Save to data/my_team.csv", disabled=True, key="save_disabled_illegal")
else:
    squad_result, squad_error = assign_squad_roles(load_round_features(round_no), selected_df)
    if squad_error:
        st.warning(squad_error)
        st.button("Save to data/my_team.csv", disabled=True, key="save_disabled_noroles")
    else:
        st.metric("Formation", squad_result.attrs["formation"])
        role_order = {"Starter": 0, "6th Man": 1, "Bench": 2}
        squad_table = squad_result.assign(
            Player=squad_result["Player"] + squad_result["captain"].map({True: " (C)", False: ""}),
            _order=squad_result["role"].map(role_order),
        ).sort_values(["_order", "predicted_score"], ascending=[True, False])
        st.dataframe(
            squad_table[["Player", "role", "position", "turn", "predicted_score"]],
            width="stretch",
        )

        # ---- Slice 7: save to data/my_team.csv (deliberate click only) ----
        st.divider()
        st.header("Save squad to data/my_team.csv")
        st.caption(
            "OVERWRITES the file the pipeline reads. First save backs the old file up "
            "to my_team.csv.bak; the write is atomic. Nothing is written until you click Save."
        )
        if coach_choice is None:
            st.info("Choose a coach to see the rows and enable saving.")
            st.button("Save to data/my_team.csv", disabled=True, key="save_disabled_nocoach")
        else:
            sheet_names = {
                r.Player_ID.strip(): f"{r.Surname.upper()}, {r.Name}"
                for r in selected_df.itertuples()
            }
            role_csv = {"Starter": "starter", "6th Man": "6th_man", "Bench": "bench"}
            save_rows = [
                {
                    "player_name": sheet_names[r.Player_ID],
                    "role": role_csv[r.role],
                    "captain": "Y" if r.captain else "",
                }
                for r in squad_result.itertuples()
            ] + [{"player_name": coach_row["coach"], "role": "coach", "captain": ""}]

            st.subheader("Preview -- exact file content that will be written")
            st.code(render_csv(save_rows, read_header_comments()), language="text")

            if st.button("Save to data/my_team.csv", type="primary"):
                try:
                    written = save_my_team(save_rows)
                    st.success(f"Saved 11 rows to {written}")
                except Exception as exc:  # show, never swallow
                    st.error(f"Save failed, my_team.csv unchanged: {exc}")
