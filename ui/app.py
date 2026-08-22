"""Streamlit app for interactively testing survivor pool algorithms.

Steps through a season week by week: the selected algorithm recommends a
pick with its win probability and reasoning, you can accept it or pick a
different available team, and confirming reveals the actual result and
locks in that week before advancing to the next.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backtest import simulator as sim  # noqa: E402
from data import nflverse_client as nc  # noqa: E402
from models import win_prob as wp  # noqa: E402
from strategy import entry_a_value  # noqa: E402
from strategy import entry_b_hedge  # noqa: E402

st.set_page_config(page_title="Survivor Picker", layout="wide")

ALGORITHM_LABELS = ["Entry A - Value Max", "Entry B - Hedge", "Baseline - Highest Win Prob"]

STATE_KEYS = (
    "week_mode_active",
    "week_mode_season",
    "week_mode_algorithm",
    "week_mode_current_week",
    "week_mode_used_teams",
    "week_mode_log",
    "week_mode_eliminated",
)


@st.cache_data(show_spinner=False)
def _available_seasons():
    return nc.get_available_seasons()


@st.cache_data(show_spinner="Loading nflverse schedule...")
def _load_schedule(season: int) -> pd.DataFrame:
    return nc.load_games(season=season)


@st.cache_resource(show_spinner="Calibrating spread model...")
def _get_spread_model():
    return wp.get_spread_model()


@dataclass(frozen=True)
class WeeklyRecommendation:
    """The algorithm's suggested pick for one week, plus the full available pool."""

    team: str
    opponent: str
    is_home: bool
    win_probability: float
    spread_line: Optional[float]
    reasoning: str
    available: List[entry_b_hedge.TeamCandidate]


def get_weekly_recommendation(
    algorithm_label: str,
    season: int,
    week: int,
    used_teams: set,
    schedule: pd.DataFrame,
    spread_model: wp.SpreadModel,
) -> Optional[WeeklyRecommendation]:
    """Recommend a pick for `week`, always backed by the full available pool.

    The dropdown needs every unused team playing this week regardless of
    whether the algorithm itself would recommend it (e.g. Entry B's floor
    can rule out every team some weeks) -- so `available` is built once,
    unfiltered, and the algorithm-specific pick/reasoning is layered on
    top of it, falling back to the closest option if the algorithm has
    nothing that meets its own criteria this week.
    """
    available = entry_b_hedge.build_candidates(
        season, week, used_teams, schedule=schedule, spread_model=spread_model
    )
    if not available:
        return None

    try:
        if algorithm_label == "Entry A - Value Max":
            rec = entry_a_value.recommend_pick(
                season, week, used_teams=used_teams, schedule=schedule, spread_model=spread_model
            )
            team, win_probability, spread_line, reasoning = (
                rec.team,
                rec.win_probability,
                rec.spread_line,
                rec.reasoning,
            )
        elif algorithm_label == "Entry B - Hedge":
            rec = entry_b_hedge.recommend_pick(
                season, week, used_teams=used_teams, schedule=schedule, spread_model=spread_model
            )
            team, win_probability, spread_line, reasoning = (
                rec.team,
                rec.win_probability,
                rec.spread_line,
                rec.reasoning,
            )
        else:  # Baseline - Highest Win Prob
            top = max(available, key=lambda c: c.win_probability)
            team, win_probability, spread_line = top.team, top.win_probability, top.spread_line
            reasoning = (
                f"{top.team} ({'vs' if top.is_home else '@'} {top.opponent}): "
                f"{top.win_probability:.1%} win probability -- the highest of any available team this week."
            )
    except ValueError as exc:
        top = max(available, key=lambda c: c.win_probability)
        team, win_probability, spread_line = top.team, top.win_probability, top.spread_line
        reasoning = f"No team meets {algorithm_label}'s criteria this week ({exc}). Showing the closest option."

    picked = next(c for c in available if c.team == team)
    return WeeklyRecommendation(
        team=team,
        opponent=picked.opponent,
        is_home=picked.is_home,
        win_probability=win_probability,
        spread_line=spread_line,
        reasoning=reasoning,
        available=available,
    )


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


def _reset_simulation(season: int, algorithm_label: str, starting_week: int) -> None:
    st.session_state["week_mode_active"] = True
    st.session_state["week_mode_season"] = season
    st.session_state["week_mode_algorithm"] = algorithm_label
    st.session_state["week_mode_current_week"] = int(starting_week)
    st.session_state["week_mode_used_teams"] = set()
    st.session_state["week_mode_log"] = []
    st.session_state["week_mode_eliminated"] = False


def _style_log_table(df: pd.DataFrame):
    def _color_result(value):
        if value == "WIN":
            return "color: #1a7f37; font-weight: 600;"
        if value in ("LOSS", "TIE"):
            return "color: #cf222e; font-weight: 600;"
        return ""

    def _color_flag(value):
        return "" if value == "Match" else "color: #9a6700; font-weight: 600;"

    return (
        df.style.map(_color_result, subset=["Result"])
        .map(_color_flag, subset=["Match/Override"])
        .format({"Spread": "{:+.1f}", "Predicted Win Probability": "{:.1%}"})
    )


def main() -> None:
    st.title("Survivor Pool Algorithm Backtester")
    st.caption("Step through a season week by week, comparing algorithm picks against real nflverse results.")

    seasons = _available_seasons()

    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col1:
        season = st.selectbox("Season", options=list(reversed(seasons)), index=0)
    with col2:
        algorithm_label = st.selectbox("Algorithm", options=ALGORITHM_LABELS)
    with col3:
        starting_week = st.number_input("Starting Week", min_value=1, max_value=22, value=1, step=1)
    with col4:
        st.write("")
        st.write("")
        if st.button("Reset Simulation", type="primary"):
            _reset_simulation(season, algorithm_label, starting_week)

    if not st.session_state.get("week_mode_active"):
        st.info("Pick a season and algorithm, then click Reset Simulation to start picking week by week.")
        return

    season = st.session_state["week_mode_season"]
    algorithm_label = st.session_state["week_mode_algorithm"]
    current_week = st.session_state["week_mode_current_week"]
    used_teams: set = st.session_state["week_mode_used_teams"]
    eliminated: bool = st.session_state["week_mode_eliminated"]
    log: list = st.session_state["week_mode_log"]

    schedule = _load_schedule(season)
    spread_model = _get_spread_model()
    max_week = int(schedule["week"].max())

    st.divider()

    if eliminated:
        st.error(f"Eliminated in Week {log[-1]['Week']}. Click Reset Simulation to start over.")
    elif current_week > max_week:
        st.success(f"Reached the end of the {season} season without losing a pick!")
    else:
        recommendation = get_weekly_recommendation(
            algorithm_label, season, current_week, used_teams, schedule, spread_model
        )
        if recommendation is None:
            st.warning(f"No available (unused) teams have a game in Week {current_week}. Stopping here.")
        else:
            st.subheader(f"Week {current_week}")
            spread_text = (
                f", spread {recommendation.spread_line:+.1f}" if recommendation.spread_line is not None else ""
            )
            st.markdown(
                f"**Algorithm recommends: {recommendation.team}** "
                f"({'vs' if recommendation.is_home else '@'} {recommendation.opponent}) "
                f"— {recommendation.win_probability:.1%} win probability{spread_text}."
            )
            st.caption(recommendation.reasoning)

            options = [c.team for c in recommendation.available]
            labels = {
                c.team: f"{c.team} ({'vs' if c.is_home else '@'} {c.opponent}, {c.win_probability:.1%})"
                for c in recommendation.available
            }
            default_index = options.index(recommendation.team) if recommendation.team in options else 0
            widget_key = f"pick_select_{season}_{algorithm_label}_{current_week}"
            selected_team = st.selectbox(
                "Your pick for this week",
                options=options,
                index=default_index,
                format_func=lambda t: labels[t],
                key=widget_key,
            )

            candidate = next(c for c in recommendation.available if c.team == selected_team)
            game_row = sim.find_game_row(schedule, current_week, selected_team)
            actual_result, outcome = sim.score_pick(game_row, selected_team)

            if outcome == "UNPLAYED":
                st.info("This matchup hasn't been played yet — check back once it's final.")
            else:
                if st.button("Confirm Pick & Advance", type="primary"):
                    used_teams.add(selected_team)
                    log.append(
                        {
                            "Week": current_week,
                            "Algorithm's Suggestion": recommendation.team,
                            "Your Actual Pick": selected_team,
                            "Predicted Win Probability": candidate.win_probability,
                            "Opponent": ("vs " if candidate.is_home else "@ ") + candidate.opponent,
                            "Spread": candidate.spread_line,
                            "Actual Score": _actual_score_display(
                                schedule, current_week, selected_team, candidate.opponent, candidate.is_home
                            ),
                            "Result": outcome,
                            "Match/Override": "Match" if selected_team == recommendation.team else "Override",
                        }
                    )
                    st.session_state["week_mode_log"] = log
                    st.session_state["week_mode_used_teams"] = used_teams
                    if outcome in ("LOSS", "TIE"):
                        st.session_state["week_mode_eliminated"] = True
                    else:
                        st.session_state["week_mode_current_week"] = current_week + 1
                    st.rerun()

    if log:
        st.divider()
        st.subheader("Results Log")
        log_df = pd.DataFrame(log)
        table_height = min(35 * (len(log_df) + 1) + 3, 800)
        st.dataframe(
            _style_log_table(log_df),
            use_container_width=True,
            hide_index=True,
            height=table_height,
        )


if __name__ == "__main__":
    main()
