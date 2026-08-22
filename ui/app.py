"""Streamlit app for testing survivor pool algorithms against a season.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backtest import simulator as sim  # noqa: E402
from data import nflverse_client as nc  # noqa: E402
from models import win_prob as wp  # noqa: E402

st.set_page_config(page_title="Survivor Picker", layout="wide")

ALGORITHMS = {
    "Entry A - Value Max": lambda schedule, spread_model: sim.make_entry_a_algorithm(
        schedule=schedule, spread_model=spread_model
    ),
    "Entry B - Hedge": lambda schedule, spread_model: sim.make_entry_b_algorithm(),
    "Baseline - Highest Win Prob": lambda schedule, spread_model: sim.highest_win_probability_algorithm,
}

STOP_REASON_LABELS = {
    "ran_out_of_available_teams": "ran out of available teams",
    "hit_unplayed_game": "reached an unplayed/future game",
}


@st.cache_data(show_spinner=False)
def _available_seasons():
    return nc.get_available_seasons()


@st.cache_data(show_spinner="Loading nflverse schedule...")
def _load_schedule(season: int) -> pd.DataFrame:
    return nc.load_games(season=season)


@st.cache_resource(show_spinner="Calibrating spread model...")
def _get_spread_model():
    return wp.get_spread_model()


def _actual_score_display(schedule: pd.DataFrame, week: int, team: str, opponent: str, is_home: bool) -> str:
    home_team = team if is_home else opponent
    away_team = opponent if is_home else team
    row = schedule[
        (schedule["week"] == week) & (schedule["home_team"] == home_team) & (schedule["away_team"] == away_team)
    ]
    if row.empty:
        return "-"
    home_score, away_score = row.iloc[0]["home_score"], row.iloc[0]["away_score"]
    if pd.isna(home_score) or pd.isna(away_score):
        return "-"
    team_score = home_score if is_home else away_score
    opp_score = away_score if is_home else home_score
    return f"{int(team_score)}-{int(opp_score)}"


def _records_to_dataframe(result: sim.BacktestResult, schedule: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "Week": r.week,
            "Pick": r.pick,
            "Opponent": ("vs " if r.is_home else "@ ") + r.opponent,
            "Predicted Win Probability %": round(r.predicted_win_probability * 100, 1),
            "Spread": r.spread_line,
            "Actual Score": _actual_score_display(schedule, r.week, r.pick, r.opponent, r.is_home),
            "Result": r.outcome,
            "Status": "Alive" if r.still_alive else "Eliminated",
        }
        for r in result.records
    ]
    return pd.DataFrame(rows)


def _style_results_table(df: pd.DataFrame):
    def _color_result(value):
        if value == "WIN":
            return "color: #1a7f37; font-weight: 600;"
        if value in ("LOSS", "TIE"):
            return "color: #cf222e; font-weight: 600;"
        return ""

    def _gray_out_eliminated(row):
        style = "opacity: 0.5;" if row["Status"] == "Eliminated" else ""
        return [style] * len(row)

    return (
        df.style.map(_color_result, subset=["Result"])
        .apply(_gray_out_eliminated, axis=1)
        .format({"Spread": "{:+.1f}", "Predicted Win Probability %": "{:.1f}"})
    )


def _summary_line(result: sim.BacktestResult) -> str:
    weeks_played = len(result.records)
    if result.eliminated_week is not None:
        return (
            f"Survived {result.weeks_survived} of {weeks_played} weeks played "
            f"— eliminated in Week {result.eliminated_week}."
        )
    if result.survived_full_season:
        return f"Survived the full season: {result.weeks_survived} of {weeks_played} weeks."
    reason = STOP_REASON_LABELS.get(result.stop_reason, result.stop_reason)
    return f"Survived {result.weeks_survived} of {weeks_played} weeks played — stopped ({reason})."


def main() -> None:
    st.title("Survivor Pool Algorithm Backtester")
    st.caption("Simulate survivor pool picks against real nflverse results.")

    seasons = _available_seasons()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        season = st.selectbox("Season", options=list(reversed(seasons)), index=0)
    with col2:
        algorithm_label = st.selectbox("Algorithm", options=list(ALGORITHMS.keys()))
    with col3:
        starting_week = st.number_input("Starting Week", min_value=1, max_value=22, value=1, step=1)

    run_clicked = st.button("Run Simulation", type="primary")

    if run_clicked:
        with st.spinner(f"Simulating '{algorithm_label}' for {season}..."):
            schedule = _load_schedule(season)
            spread_model = _get_spread_model()
            algorithm = ALGORITHMS[algorithm_label](schedule, spread_model)
            result = sim.simulate(
                season,
                int(starting_week),
                algorithm,
                algorithm_name=algorithm_label,
                schedule=schedule,
                spread_model=spread_model,
            )
        st.session_state["result"] = result
        st.session_state["schedule"] = schedule

    result: sim.BacktestResult | None = st.session_state.get("result")
    schedule: pd.DataFrame | None = st.session_state.get("schedule")

    if result is None:
        st.info("Pick a season and algorithm, then click Run Simulation.")
        return

    st.subheader(_summary_line(result))

    if not result.records:
        st.warning("No weeks were simulated for this selection.")
        return

    chart_df = pd.DataFrame(
        {
            "Week": [r.week for r in result.records],
            "Predicted Win Probability": [r.predicted_win_probability for r in result.records],
        }
    ).set_index("Week")
    st.line_chart(chart_df, y="Predicted Win Probability")

    table_df = _records_to_dataframe(result, schedule)
    table_height = min(35 * (len(table_df) + 1) + 3, 800)
    st.dataframe(
        _style_results_table(table_df),
        use_container_width=True,
        hide_index=True,
        height=table_height,
    )


if __name__ == "__main__":
    main()
