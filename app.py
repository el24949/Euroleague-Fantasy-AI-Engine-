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

from pathlib import Path

import pandas as pd
import streamlit as st

from data_collection.load_prices_history import load_prices_for_round
from features.turn_tags import build_turn_lookup
from my_team_writer import read_header_comments, render_csv, save_my_team
from optimizer.my_team import (
    POSITION_NAME_TO_CODE,
    PRICE_FILE,
    RECOVERY_SEASON,
    SEASON as LIVE_SEASON,
    load_boxscore_players,
    load_coach_pool,
    load_my_team,
    load_prices,
    load_price_pool,
    resolve_coach,
    resolve_player_ids,
)
from optimizer.pick_lineup_ilp import RISK_LAMBDA, solve_lineup
from optimizer.pick_team_ilp import BUDGET, MAX_PER_CLUB
from optimizer.suggest_trades import suggest_trades
from optimizer.turn_join import attach_turns
from pipeline.update_round import (
    SEASON,
    get_recommended_team,
    get_round_gamecodes,
    run_prediction_step,
    status_summary,
)

st.set_page_config(page_title="EuroLeague Fantasy -- My Team", layout="wide")

# Minimal CSS: Inter font + a bit of spacing + orange section headers, nothing
# else -- no custom card components, backgrounds stay in .streamlit/config.toml's
# theme. Header orange = the theme's primaryColor, the single accent color.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    .block-container { padding-top: 2.5rem; }
    div[data-testid="stHeading"] h1,
    div[data-testid="stHeading"] h2,
    div[data-testid="stHeading"] h3 { color: #FF6B1A; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROLE_LABELS = {"starter": "Starters", "6th_man": "6th Man", "bench": "Bench"}
ROLE_ORDER = ["starter", "6th_man", "bench"]


@st.cache_data
def load_my_team_cached():
    """load_my_team() (optimizer/my_team.py) behind a cached function, same
    pattern as load_player_table/load_coach_table -- it parses the price
    Excel sheet and queries the DB, so it shouldn't rerun on every widget
    interaction. Cleared explicitly after a successful save (see the Save
    section below) so this view reflects the roster just written, not a
    stale cached read of the file from before the save."""
    return load_my_team()


@st.cache_data
def get_next_round_cached(season):
    """status_summary() (pipeline/update_round.py) behind a cached function --
    reuses its Schedule-based round<->gamecode mapping (never a hardcoded
    "round N = gamecodes ..." formula) to find the first round with any
    scheduled game missing from the boxscores table, i.e. the next round to
    pick a team for. Only next_round is used here; status_summary's other
    stats and its own prints (to the server console, not this page) aren't
    needed."""
    return status_summary(season)["next_round"]


def render_role_table(players, role):
    """One role group's players as a clean Player/Position/Price table --
    Player_ID and the raw role code are never shown. The captain's name gets
    a '(C)' suffix; nothing else is computed here beyond that display label."""
    group = players[players["role"] == role]
    if group.empty:
        return
    st.markdown(f"**{ROLE_LABELS[role]}**")
    display = pd.DataFrame(
        {
            "Player": group["player_name"] + group["captain"].apply(
                lambda v: " (C)" if v == "Y" else ""
            ),
            "Position": group["position"],
            "Price": group["price"],
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)


REQUIRED_POSITION_COUNTS = {"G": 4, "F": 4, "C": 2}


@st.cache_data
def load_player_table():
    """optimizer.my_team.load_prices() behind a cached function, same pattern as
    load_recommendation -- it parses an Excel file, so it shouldn't run on every
    rerun. load_prices() returns (df, meta_dict) of excluded rows (head coaches,
    missing position, placeholder ID); this slice only needs the usable df, so
    meta_dict is discarded here and never returned."""
    df, _meta = load_prices()
    return df


@st.cache_data
def load_round_features_for(season, round_no):
    """Same reuse as load_round_features below (run_prediction_step() behind a
    cache), just parameterized by season too -- needed because Suggest Trades
    (and Apply these trades, which re-assigns roles for the same season/round
    the suggestion was computed for) can target Season 2025 or 2026, not only
    pipeline/update_round.py's own module-level SEASON=2025 the rest of this
    page's "Recommended Team"/"Squad roles" sections use."""
    return run_prediction_step(get_round_gamecodes(season, round_no))


def assign_squad_roles(round_features, selected_df):
    """Slice 6: run the existing solve_lineup() with the roster FIXED to the
    selected squad. Returns (result_df, error_message); exactly one is None.

    The candidate pool is restricted to the selected Player_IDs AND those rows
    are forced onto the roster (fixed_roster_idx), so only roles + formation are
    optimized. Formation/weights/risk objective come from pick_lineup_ilp
    unchanged. Captain is NOT decided by the ILP (captain_multiplier=1.0 zeroes
    its bonus and its captain pick is ignored): it is a post-step over T1
    starters only. Prices are the live price table's, same as the legality check.

    Moved up from its original position further down this file (same body,
    unchanged) so the Suggest Trades / Apply These Trades section above can
    call it too -- Streamlit executes this script top-to-bottom, so a function
    must be defined before the first section that uses it."""
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


# ---- Upload round prices ----
# Saves the uploaded file under the exact name pattern
# data_collection.latest_prices.find_latest_price_file expects
# (players_stats26_<round>.xlsx) so the engine picks it up as the latest
# round's prices on its next run/rerun. This section only writes the file --
# it never reloads PRICE_FILE, re-imports optimizer.my_team, or recomputes
# anything.
UPLOADS_DIR = Path("data/prices/uploads")

st.header("Upload round prices")
st.caption(
    "Upload this round's players_stats Excel file. Once saved, the engine will "
    "use it as the latest round's prices on its next run/rerun."
)

uploaded_price_file = st.file_uploader(
    "Price file (.xlsx)", type=["xlsx"], key="price_upload_file"
)
upload_round_no = st.number_input(
    "Which round is this price file for?",
    min_value=1,
    step=1,
    value=None,
    format="%d",
    key="price_upload_round",
)

if st.button("Save price file"):
    if uploaded_price_file is None:
        st.error("Choose a .xlsx price file first.")
    elif upload_round_no is None:
        st.error("Enter which round this price file is for.")
    elif not uploaded_price_file.name.lower().endswith(".xlsx"):
        st.error("Only .xlsx files are accepted.")
    else:
        target_path = UPLOADS_DIR / f"players_stats26_{int(upload_round_no)}.xlsx"
        if target_path.exists():
            # Don't silently overwrite -- stage it and require a second,
            # explicit confirmation click below.
            st.session_state["price_upload_pending"] = {
                "bytes": uploaded_price_file.getvalue(),
                "path": str(target_path),
                "round": int(upload_round_no),
            }
        else:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(uploaded_price_file.getvalue())
            st.success(
                f"Saved to {target_path} (round {int(upload_round_no)}). "
                "The engine will use this as the latest round's prices on its next run."
            )

pending_upload = st.session_state.get("price_upload_pending")
if pending_upload:
    st.warning(
        f"A price file for round {pending_upload['round']} already exists at "
        f"{pending_upload['path']}. Overwrite it?"
    )
    confirm_cols = st.columns(2)
    if confirm_cols[0].button("Yes, overwrite", type="primary", key="confirm_price_overwrite"):
        Path(pending_upload["path"]).write_bytes(pending_upload["bytes"])
        st.success(
            f"Overwrote {pending_upload['path']} (round {pending_upload['round']}). "
            "The engine will use this as the latest round's prices on its next run."
        )
        del st.session_state["price_upload_pending"]
    if confirm_cols[1].button("Cancel", key="cancel_price_overwrite"):
        del st.session_state["price_upload_pending"]

st.divider()

st.title("My Team")

try:
    my_team = load_my_team_cached()
except ValueError as exc:
    st.error(f"Couldn't load data/my_team.csv: {exc}")
else:
    my_players = my_team["players"]
    my_coach = my_team["coach"]
    my_state = my_team["state"]
    squad_cost = float(my_players["price"].sum()) + float(my_coach["price"])

    metric_cols = st.columns(3)
    metric_cols[0].metric("Credits left", f"{my_state['remaining_budget']:.1f}")
    metric_cols[1].metric("Trades used", my_state["trades_used"])
    metric_cols[2].metric("Squad cost", f"{squad_cost:.1f} / 100")
    if squad_cost > 100:
        metric_cols[2].markdown(":red[Over budget]")

    st.subheader("My Team")
    for role in ROLE_ORDER:
        render_role_table(my_players, role)

    st.subheader("Coach")
    coach_cols = st.columns(3)
    coach_cols[0].metric("Name", my_coach["name"])
    coach_cols[1].metric("Team", my_coach["team_code"])
    coach_cols[2].metric("Price", f"{my_coach['price']:.1f}")

st.divider()
st.header("Suggest trades")
st.caption(
    "Runs the existing trade-suggestion optimizer (optimizer/suggest_trades.py) "
    "for the round below. Display only -- nothing is saved to data/my_team.csv "
    "in this step."
)

trade_season = st.selectbox(
    "Season",
    options=[2026, 2025],
    index=[2026, 2025].index(LIVE_SEASON),
    key="trade_season",
)
st.caption(
    "2026 is the live season; 2025 has full history and is useful for testing "
    "the button."
)

auto_next_round = get_next_round_cached(trade_season)
if auto_next_round is None:
    st.warning(
        f"Couldn't auto-detect the next round for Season {trade_season} -- every "
        "scheduled round already has box scores. Pick a round number manually below."
    )
    default_round = 1
else:
    st.caption(f"Auto-detected next round: {auto_next_round} (last round with box scores + 1).")
    default_round = auto_next_round

target_round = st.number_input(
    "Predict for round",
    min_value=1,
    step=1,
    value=default_round,
    key="trade_target_round",
)

trade_input_cols = st.columns(2)
unlimited_trades = trade_input_cols[0].checkbox("Unlimited trades", key="unlimited_trades")
max_trades_input = trade_input_cols[1].number_input(
    "Max trades this round",
    min_value=0,
    step=1,
    value=4,
    disabled=unlimited_trades,
    key="max_trades_input",
)
st.caption("Unlimited-trade rounds this season: after R6, R13, R18, R23, R28, R34.")

if st.button("Suggest trades", type="primary"):
    try:
        my_team_for_trades = load_my_team_cached()
    except ValueError as exc:
        st.error(f"Couldn't load data/my_team.csv: {exc}")
    else:
        max_trades = None if unlimited_trades else int(max_trades_input)
        try:
            trade_result = suggest_trades(
                my_team_for_trades,
                max_trades,
                season=trade_season,
                round_no=int(target_round),
                budget=BUDGET,
            )
        except Exception as exc:  # e.g. no scheduled games, or no prior-history
            # data yet for this round -- show it, never crash the page.
            st.error(f"Couldn't compute trade suggestions for round {int(target_round)}: {exc}")
        else:
            # Stashed in session_state (not just a local variable) because
            # Streamlit reruns this whole script on every interaction -- the
            # "Apply these trades" button below needs this SAME result on a
            # LATER rerun, not a recompute (suggest_trades can be slow, and
            # re-running it could legitimately return a different answer).
            st.session_state["trade_result"] = trade_result
            st.session_state["trade_context"] = {
                "season": trade_season,
                "round_no": int(target_round),
            }
            st.session_state.pop("apply_trades_pending", None)  # stale preview

trade_result = st.session_state.get("trade_result")
if trade_result is not None and trade_result["swaps"] and "out_position" not in trade_result["swaps"][0]:
    # A result cached in session_state from BEFORE a swap-dict schema change
    # (e.g. this app was edited and hot-reloaded mid-session) -- discard it
    # instead of KeyError-ing on a since-renamed field. Confirmed reproducible:
    # an old-shaped {"position": ...} dict run through today's rename/column
    # selection raises exactly "['Out pos', 'In pos'] not in index".
    st.session_state.pop("trade_result", None)
    st.session_state.pop("trade_context", None)
    st.session_state.pop("apply_trades_pending", None)
    trade_result = None
    st.info("Trade suggestions were cleared after an app update -- click 'Suggest trades' again.")

if trade_result is not None:
    score_cols = st.columns(2)
    score_cols[0].metric("Current projected score", f"{trade_result['current_score']:.2f}")
    score_cols[1].metric(
        "Projected score after trades",
        f"{trade_result['projected_score']:.2f}",
        delta=f"{trade_result['projected_score'] - trade_result['current_score']:+.2f}",
    )

    trades_label = (
        f"{trade_result['trades_used']} / unlimited"
        if trade_result["max_trades"] is None
        else f"{trade_result['trades_used']} / {trade_result['max_trades']}"
    )
    cost_cols = st.columns(2)
    cost_cols[0].metric("Trades used", trades_label)
    cost_cols[1].metric(
        "Resulting total cost", f"{trade_result['resulting_total_cost']:.1f} / {BUDGET}"
    )
    if trade_result["resulting_total_cost"] > BUDGET:
        cost_cols[1].markdown(":red[Over budget]")

    if trade_result["swaps"]:
        st.subheader("Suggested swaps")
        swap_table = pd.DataFrame(trade_result["swaps"]).rename(
            columns={
                "out_player": "Out",
                "out_price": "Out price",
                "out_position": "Out pos",
                "in_player": "In",
                "in_price": "In price",
                "in_position": "In pos",
                "projected_gain": "Gain",
            }
        )[["Out", "Out pos", "Out price", "In", "In pos", "In price", "Gain"]]
        st.dataframe(
            swap_table.style.map(
                lambda v: f"color: {'green' if v >= 0 else 'red'}", subset=["Gain"]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("No trades suggested -- your team is already optimal within the limit.")

    coach_change = trade_result["coach_change"]
    if coach_change["changed"]:
        st.write(f"Coach change: {coach_change['from']} -> {coach_change['to']}")
    else:
        st.write(f"Coach unchanged: {coach_change['from']}")

    # ---- Apply these trades: write the new roster to data/my_team.csv ----
    st.subheader("Apply these trades")
    st.caption(
        "Builds the new 10-player squad from the swaps above, re-assigns roles "
        "with the same optimizer the Squad roles section below uses, and "
        "OVERWRITES data/my_team.csv. First save backs up the old file to "
        "my_team.csv.bak; the write is atomic. Does NOT touch "
        "data/my_team_state.csv -- trades_used is still yours to update."
    )

    if not trade_result["swaps"] and not coach_change["changed"]:
        st.caption("Nothing to apply -- no trades were suggested.")
    else:
        if st.button("Apply these trades", key="apply_trades_button"):
            try:
                current_for_apply = load_my_team_cached()
                trade_ctx = st.session_state["trade_context"]
                # load_price_pool() (the live PRICE_FILE) -- NOT load_player_table()
                # (load_prices() with no path, which defaults to the older generic
                # data/prices/players_stats.xlsx). suggest_trades.py's own position
                # constraint was solved against load_price_pool(); rebuilding the
                # squad from a DIFFERENT price file risks a player having a
                # different Position/Team there than what the optimizer actually
                # used, breaking the 4G/4F/2C shape it already guaranteed.
                price_pool_for_apply = load_price_pool()
                usable_pool_for_apply = price_pool_for_apply[
                    ~price_pool_for_apply["is_placeholder"]
                ]

                out_names = {s["out_player"] for s in trade_result["swaps"]}
                kept = current_for_apply["players"][
                    ~current_for_apply["players"]["player_name"].isin(out_names)
                ]
                kept_ids = kept["Player_ID"].tolist()

                in_names = [s["in_player"] for s in trade_result["swaps"]]
                if in_names:
                    # resolve_player_ids is the SAME name->Player_ID resolver
                    # load_my_team() itself uses -- fails loudly on a zero/
                    # ambiguous match rather than guessing.
                    resolved_in = resolve_player_ids(
                        in_names, price_pool_for_apply, load_boxscore_players(RECOVERY_SEASON)
                    )
                    new_ids = resolved_in["Player_ID"].tolist()
                else:
                    new_ids = []

                new_squad_df = usable_pool_for_apply[
                    usable_pool_for_apply["Player_ID"].isin(kept_ids + new_ids)
                ].copy()
                if len(new_squad_df) != 10:
                    raise ValueError(
                        f"Expected 10 players after applying trades, found "
                        f"{len(new_squad_df)} -- a swapped-in player may not be a "
                        "usable row in the current price file."
                    )

                if coach_change["changed"]:
                    coach_match = resolve_coach(coach_change["to"], load_coach_pool())
                    new_coach = {
                        "name": coach_match["coach"],
                        "team_code": coach_match["team_code"],
                        "price": coach_match["coach_price"],
                    }
                else:
                    new_coach = current_for_apply["coach"]

                round_features_for_apply = load_round_features_for(
                    trade_ctx["season"], trade_ctx["round_no"]
                )
                new_squad_result, new_squad_error = assign_squad_roles(
                    round_features_for_apply, new_squad_df
                )
            except Exception as exc:
                st.error(f"Couldn't apply trades: {exc}")
            else:
                if new_squad_error:
                    st.error(f"Couldn't assign roles for the new squad: {new_squad_error}")
                else:
                    sheet_names = {
                        r.Player_ID.strip(): f"{r.Surname.upper()}, {r.Name}"
                        for r in new_squad_df.itertuples()
                    }
                    role_csv = {"Starter": "starter", "6th Man": "6th_man", "Bench": "bench"}
                    apply_save_rows = [
                        {
                            "player_name": sheet_names[r.Player_ID],
                            "role": role_csv[r.role],
                            "captain": "Y" if r.captain else "",
                        }
                        for r in new_squad_result.itertuples()
                    ] + [{"player_name": new_coach["name"], "role": "coach", "captain": ""}]

                    # Staged, not written yet -- see the "Confirm save" button
                    # below. Same two-click pattern as the price-upload
                    # overwrite guard above: nothing touches disk on this click.
                    st.session_state["apply_trades_pending"] = {
                        "save_rows": apply_save_rows,
                        "preview_text": render_csv(apply_save_rows, read_header_comments()),
                    }

        pending_apply = st.session_state.get("apply_trades_pending")
        if pending_apply:
            st.subheader("Preview -- exact file content that will be written")
            st.code(pending_apply["preview_text"], language="text")
            apply_confirm_cols = st.columns(2)
            if apply_confirm_cols[0].button(
                "Confirm save", type="primary", key="confirm_apply_trades"
            ):
                try:
                    written = save_my_team(pending_apply["save_rows"])
                except Exception as exc:
                    st.error(f"Save failed, my_team.csv unchanged: {exc}")
                else:
                    load_my_team_cached.clear()  # so "My Team" above reflects this save
                    st.success(f"Saved 11 rows to {written}")
                    del st.session_state["apply_trades_pending"]
                    # The applied suggestion is now stale against the new roster.
                    st.session_state.pop("trade_result", None)
                    st.session_state.pop("trade_context", None)
            if apply_confirm_cols[1].button("Cancel", key="cancel_apply_trades"):
                del st.session_state["apply_trades_pending"]

st.divider()

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


@st.cache_data
def load_coach_table():
    """optimizer.my_team.load_coach_pool(PRICE_FILE) behind a cached function: the
    head coaches (name + price) from the SAME round_01 price file load_my_team()
    reads them from, so the app's coach price and the loader's are one value.
    Sorted by price (desc) for a stable picker order."""
    return load_coach_pool(PRICE_FILE).sort_values("coach_price", ascending=False).to_dict("records")


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
# Coaches + prices come from PRICE_FILE (round_01), the same file load_my_team()
# uses -- NOT the 2025 round shortlist above, whose prices are from an older sheet.
coach_pool = load_coach_table()
coach_choice = st.selectbox(
    f"Coach (from {PRICE_FILE})",
    options=range(len(coach_pool)),
    index=None,
    placeholder="Pick a coach (its price counts toward the 100 budget)",
    format_func=lambda i: (
        f"{coach_pool[i]['coach']} ({coach_pool[i]['team_code']}) -- "
        f"price {coach_pool[i]['coach_price']:.1f}"
    ),
)
coach_row = coach_pool[coach_choice] if coach_choice is not None else None


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

POSITION_LABELS = {"G": "Guards", "F": "Forwards", "C": "Centers"}
SQUAD_SIZE = 10

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
                    load_my_team_cached.clear()  # so the My Team view above reflects this save
                    st.success(f"Saved 11 rows to {written}")
                except Exception as exc:  # show, never swallow
                    st.error(f"Save failed, my_team.csv unchanged: {exc}")
