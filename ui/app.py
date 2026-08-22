"""Streamlit app for interactively testing the survivor pool strategies.

Steps through a season week by week for both entries simultaneously:
Entry A always picks first using its own value-max algorithm (independent
of Entry B); Entry B then picks the best team from what's left *excluding
whichever team Entry A just picked*, so the two entries never end up on
the same team. Once Entry A is eliminated, Entry B is no longer
constrained and picks freely on its own.

Each week you can accept either recommendation or override it from a
dropdown of that entry's available teams; confirming both locks them in,
reveals the actual results, and appends one row (with both entries' data
in separate columns) to a growing results log.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backtest import simulator as sim  # noqa: E402
from data import nflverse_client as nc  # noqa: E402
from models import dp_optimizer  # noqa: E402
from models import win_prob as wp  # noqa: E402
from strategy import draft_order  # noqa: E402
from strategy import entry_a_value  # noqa: E402
from strategy import entry_b_hedge  # noqa: E402

st.set_page_config(page_title="Survivor Picker", layout="wide")

ROW_COLUMNS = ["Suggestion", "Pick", "Win Prob", "Spread", "Score", "Result", "Match/Override"]


def _matchup_display(team: str, opponent: str, is_home: bool) -> str:
    """Format as AWAY@HOME, e.g. "CIN@NE" -- the second team listed is the home team."""
    away, home = (opponent, team) if is_home else (team, opponent)
    return f"{away}@{home}"


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
    """One entry's suggested pick for one week, plus its full available pool."""

    team: str
    opponent: str
    is_home: bool
    win_probability: float
    spread_line: Optional[float]
    reasoning: str
    available: List[entry_b_hedge.TeamCandidate]
    projected_path: Optional[Sequence[dp_optimizer.WeekPick]] = None


def get_entry_recommendation(
    entry: str,
    season: int,
    week: int,
    used_teams: Set[str],
    schedule: pd.DataFrame,
    spread_model: wp.SpreadModel,
    exclude_teams: Set[str] = frozenset(),
) -> Optional[WeeklyRecommendation]:
    """Recommend a pick for `entry` ("A" or "B"), excluding `exclude_teams` from the pool.

    `exclude_teams` is how Entry B is kept off whatever team Entry A picked
    this week -- Entry A itself is never called with an exclusion, since it
    always picks independently and has priority.
    """
    projected_path = None
    if entry == "A":
        raw_available = entry_a_value.build_candidates(
            season, week, used_teams, schedule=schedule, spread_model=spread_model
        )
        available = [c for c in raw_available if c.team not in exclude_teams]
        if not available:
            return None
        try:
            rec = entry_a_value.recommend_pick(
                season, week, used_teams=used_teams, schedule=schedule, spread_model=spread_model
            )
        except ValueError:
            rec = None
        if rec is not None and rec.team not in exclude_teams:
            top = next(c for c in available if c.team == rec.team)
            reasoning = rec.reasoning
            projected_path = rec.projected_path
        else:
            top = max(available, key=lambda c: c.win_probability)
            reasoning = (
                "The multi-week optimizer couldn't produce a valid plan this week "
                "(likely due to Entry B's exclusion); showing the single best option."
            )
    else:
        raw_available = entry_b_hedge.build_candidates(
            season, week, used_teams, schedule=schedule, spread_model=spread_model
        )
        available = [c for c in raw_available if c.team not in exclude_teams]
        if not available:
            return None
        eligible = entry_b_hedge.rank_picks(available)
        if eligible:
            top, runner_up = eligible[0], (eligible[1] if len(eligible) > 1 else None)
            reasoning = entry_b_hedge.build_reasoning(top, runner_up)
        else:
            top = max(available, key=lambda c: c.win_probability)
            reasoning = (
                f"No team clears the {entry_b_hedge.DEFAULT_MIN_WIN_PROBABILITY:.0%} floor "
                "this week. Showing the closest option."
            )

    return WeeklyRecommendation(
        team=top.team,
        opponent=top.opponent,
        is_home=top.is_home,
        win_probability=top.win_probability,
        spread_line=top.spread_line,
        reasoning=reasoning,
        available=available,
        projected_path=projected_path,
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


def _reset_simulation(season: int, starting_week: int) -> None:
    st.session_state["dual_active"] = True
    st.session_state["dual_season"] = season
    st.session_state["dual_current_week"] = int(starting_week)
    st.session_state["dual_used_a"] = set()
    st.session_state["dual_used_b"] = set()
    st.session_state["dual_eliminated_a"] = False
    st.session_state["dual_eliminated_b"] = False
    st.session_state["dual_log"] = []


def _render_entry_column(
    label: str,
    entry: str,
    eliminated: bool,
    recommendation: Optional[WeeklyRecommendation],
    widget_key: str,
) -> Optional[str]:
    """Render one entry's recommendation + pick dropdown; return the selected team (or None)."""
    st.markdown(f"### {label}")
    if eliminated:
        st.caption("Eliminated — no longer picking.")
        return None
    if recommendation is None:
        st.warning("No available team for this entry this week.")
        return None

    spread_text = f", spread {recommendation.spread_line:+.1f}" if recommendation.spread_line is not None else ""
    st.markdown(
        f"**Recommends: {recommendation.team}** "
        f"({'vs' if recommendation.is_home else '@'} {recommendation.opponent}) "
        f"— {recommendation.win_probability:.1%}{spread_text}"
    )
    st.caption(recommendation.reasoning)

    if recommendation.projected_path and len(recommendation.projected_path) > 1:
        with st.expander(f"Projected {len(recommendation.projected_path)}-week plan"):
            plan_df = pd.DataFrame(
                [
                    {
                        "Week": p.week,
                        "Team": p.team,
                        "Matchup": _matchup_display(p.team, p.opponent, p.is_home),
                        "Win Prob": f"{p.win_probability:.1%}",
                    }
                    for p in recommendation.projected_path
                ]
            )
            st.dataframe(plan_df, hide_index=True, use_container_width=True)

    options = [c.team for c in recommendation.available]
    labels = {
        c.team: f"{c.team} ({'vs' if c.is_home else '@'} {c.opponent}, {c.win_probability:.1%})"
        for c in recommendation.available
    }
    default_index = options.index(recommendation.team) if recommendation.team in options else 0
    return st.selectbox(
        f"{label} pick",
        options=options,
        index=default_index,
        format_func=lambda t: labels[t],
        key=widget_key,
    )


def _style_log_table(df: pd.DataFrame):
    def _fmt_pct(v):
        return f"{v:.1%}" if isinstance(v, (int, float)) else v

    def _fmt_spread(v):
        return f"{v:+.1f}" if isinstance(v, (int, float)) else v

    def _color_result(value):
        if value == "WIN":
            return "color: #1a7f37; font-weight: 600;"
        if value in ("LOSS", "TIE"):
            return "color: #cf222e; font-weight: 600;"
        return ""

    def _color_flag(value):
        return "color: #9a6700; font-weight: 600;" if value == "Override" else ""

    result_cols = [c for c in df.columns if c.endswith("Result")]
    flag_cols = [c for c in df.columns if c.endswith("Match/Override")]
    win_prob_cols = [c for c in df.columns if c.endswith("Win Prob")]
    spread_cols = [c for c in df.columns if c.endswith("Spread")]

    format_map = {c: _fmt_pct for c in win_prob_cols}
    format_map.update({c: _fmt_spread for c in spread_cols})

    styler = df.style
    if result_cols:
        styler = styler.map(_color_result, subset=result_cols)
    if flag_cols:
        styler = styler.map(_color_flag, subset=flag_cols)
    if format_map:
        styler = styler.format(format_map)
    return styler


def main() -> None:
    st.title("Survivor Pool Algorithm Backtester")
    st.caption(
        "Step through a season week by week for both entries at once: "
        "Entry A picks first, Entry B picks the best of what's left."
    )

    seasons = _available_seasons()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        season = st.selectbox("Season", options=list(reversed(seasons)), index=0)
    with col2:
        starting_week = st.number_input("Starting Week", min_value=1, max_value=22, value=1, step=1)
    with col3:
        st.write("")
        st.write("")
        if st.button("Reset Simulation", type="primary"):
            _reset_simulation(season, starting_week)

    if not st.session_state.get("dual_active"):
        st.info("Pick a season, then click Reset Simulation to start picking week by week for both entries.")
        return

    season = st.session_state["dual_season"]
    current_week = st.session_state["dual_current_week"]
    used_a: Set[str] = st.session_state["dual_used_a"]
    used_b: Set[str] = st.session_state["dual_used_b"]
    eliminated_a: bool = st.session_state["dual_eliminated_a"]
    eliminated_b: bool = st.session_state["dual_eliminated_b"]
    log: list = st.session_state["dual_log"]

    schedule = _load_schedule(season)
    spread_model = _get_spread_model()
    max_week = int(schedule["week"].max())

    st.divider()

    if eliminated_a and eliminated_b:
        st.error("Both entries have been eliminated. Click Reset Simulation to start over.")
    elif current_week > max_week:
        st.success(f"Reached the end of the {season} season!")
    else:
        st.subheader(f"Week {current_week}")

        rec_a = (
            None
            if eliminated_a
            else get_entry_recommendation("A", season, current_week, used_a, schedule, spread_model)
        )
        col_a, col_b = st.columns(2)
        with col_a:
            selected_a = _render_entry_column(
                "Entry A", "A", eliminated_a, rec_a, f"pick_a_{season}_{current_week}"
            )

        exclude_for_b = {selected_a} if selected_a else set()
        rec_b = (
            None
            if eliminated_b
            else get_entry_recommendation(
                "B", season, current_week, used_b, schedule, spread_model, exclude_teams=exclude_for_b
            )
        )
        with col_b:
            # Key includes selected_a so Entry B's widget resets cleanly whenever
            # Entry A's pick changes and its option list shifts underneath it.
            selected_b = _render_entry_column(
                "Entry B", "B", eliminated_b, rec_b, f"pick_b_{season}_{current_week}_{selected_a}"
            )

        if not eliminated_a and not eliminated_b:
            with st.expander("Top picks (draft order): Entry A's picks, then Entry B's, all distinct"):
                try:
                    draft = draft_order.draft_picks(
                        season, current_week, used_a, used_b, rounds=2, schedule=schedule, spread_model=spread_model
                    )
                except ValueError as exc:
                    st.caption(f"Couldn't compute a full draft order this week: {exc}")
                else:
                    draft_df = pd.DataFrame(
                        [
                            {
                                "Pick": d.pick_number,
                                "Entry": d.entry,
                                "Team": d.team,
                                "Matchup": _matchup_display(d.team, d.opponent, d.is_home),
                                "Win Prob": f"{d.win_probability:.1%}",
                            }
                            for d in draft
                        ]
                    )
                    st.dataframe(draft_df, hide_index=True, use_container_width=True)
                    st.caption(
                        "Each entry's later picks are computed after excluding every team already "
                        "drafted (by either entry) earlier in this order -- so a later pick can differ "
                        "from what that entry would've picked on its own."
                    )

        pending_results = []
        for selected, entry_label in ((selected_a, "A"), (selected_b, "B")):
            if selected is None:
                continue
            game_row = sim.find_game_row(schedule, current_week, selected)
            actual_result, outcome = sim.score_pick(game_row, selected)
            pending_results.append((entry_label, outcome))

        any_unplayed = any(outcome == "UNPLAYED" for _, outcome in pending_results)
        if any_unplayed:
            st.info("At least one of this week's matchups hasn't been played yet — check back once it's final.")
        elif selected_a is None and selected_b is None:
            st.warning("Neither entry has an available pick this week.")
        else:
            if st.button("Confirm Both Picks & Advance", type="primary"):
                row = {"Week": current_week}

                if selected_a is not None:
                    cand_a = next(c for c in rec_a.available if c.team == selected_a)
                    game_row_a = sim.find_game_row(schedule, current_week, selected_a)
                    _, outcome_a = sim.score_pick(game_row_a, selected_a)
                    used_a.add(selected_a)
                    row.update(
                        {
                            "A: Suggestion": _matchup_display(rec_a.team, rec_a.opponent, rec_a.is_home),
                            "A: Pick": selected_a,
                            "A: Win Prob": cand_a.win_probability,
                            "A: Spread": cand_a.spread_line,
                            "A: Score": _actual_score_display(
                                schedule, current_week, selected_a, cand_a.opponent, cand_a.is_home
                            ),
                            "A: Result": outcome_a,
                            "A: Match/Override": "Match" if selected_a == rec_a.team else "Override",
                        }
                    )
                    if outcome_a in ("LOSS", "TIE"):
                        st.session_state["dual_eliminated_a"] = True
                else:
                    row.update({f"A: {c}": "—" for c in ROW_COLUMNS})

                if selected_b is not None:
                    cand_b = next(c for c in rec_b.available if c.team == selected_b)
                    game_row_b = sim.find_game_row(schedule, current_week, selected_b)
                    _, outcome_b = sim.score_pick(game_row_b, selected_b)
                    used_b.add(selected_b)
                    row.update(
                        {
                            "B: Suggestion": _matchup_display(rec_b.team, rec_b.opponent, rec_b.is_home),
                            "B: Pick": selected_b,
                            "B: Win Prob": cand_b.win_probability,
                            "B: Spread": cand_b.spread_line,
                            "B: Score": _actual_score_display(
                                schedule, current_week, selected_b, cand_b.opponent, cand_b.is_home
                            ),
                            "B: Result": outcome_b,
                            "B: Match/Override": "Match" if selected_b == rec_b.team else "Override",
                        }
                    )
                    if outcome_b in ("LOSS", "TIE"):
                        st.session_state["dual_eliminated_b"] = True
                else:
                    row.update({f"B: {c}": "—" for c in ROW_COLUMNS})

                log.append(row)
                st.session_state["dual_log"] = log
                st.session_state["dual_used_a"] = used_a
                st.session_state["dual_used_b"] = used_b
                st.session_state["dual_current_week"] = current_week + 1
                st.rerun()

    if log:
        st.divider()
        st.subheader("Results Log")

        chart_df = pd.DataFrame(
            [{"Week": r["Week"], "Entry A": r.get("A: Win Prob"), "Entry B": r.get("B: Win Prob")} for r in log]
        ).set_index("Week")
        chart_df = chart_df.apply(pd.to_numeric, errors="coerce")
        st.line_chart(chart_df)

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
