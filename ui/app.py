"""Streamlit app for interactively testing the survivor pool strategies.

Steps through a season week by week for both entries simultaneously.
Both entries are advised by the same DP-optimizer algorithm (see
strategy.draft_order), each against its own independent used-teams
history: Entry A picks first; Entry B then picks the best team from its
own pool *excluding whichever team Entry A just picked*, so the two
entries never end up on the same team. Once Entry A is eliminated,
Entry B is no longer constrained and picks freely on its own.

Each entry's card also shows its own alternate pick (from the draft
order, see strategy.draft_order) -- Entry A's pick #2, Entry B's
pick #4 -- guaranteed distinct from every other pick shown that week.
Once one entry is eliminated, the draft keeps drafting from the
surviving entry's remaining teams, so its card shows all four
candidates instead of just its own two.

Each week you can accept a recommendation or override it from a
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
from data import nfelo_client  # noqa: E402
from data import nflverse_client as nc  # noqa: E402
from models import dp_optimizer  # noqa: E402
from models import win_prob as wp  # noqa: E402
from strategy import draft_order  # noqa: E402
from strategy import entry_a_value  # noqa: E402
from strategy import entry_b_hedge  # noqa: E402

st.set_page_config(page_title="Survivor Picker", layout="wide")

ROW_COLUMNS = [
    "Suggestion", "Pick", "Win Prob", "Spread", "Score", "Result", "Match/Override",
    "Model Divergence", "Team Bias",
]

# Market/Elo blend options offered in the UI, in market_weight order (see
# models.win_prob.get_win_probability). Label -> market_weight.
MARKET_WEIGHT_OPTIONS = {
    "100% Market / 0% Elo": 1.0,
    "75% Market / 25% Elo": 0.75,
    "50% Market / 50% Elo": 0.5,
    "25% Market / 75% Elo": 0.25,
    "0% Market / 100% Elo": 0.0,
}
DEFAULT_MARKET_WEIGHT_LABEL = "75% Market / 25% Elo"


def _matchup_display(team: str, opponent: str, is_home: bool, bold_team: bool = False) -> str:
    """Format as AWAY@HOME, e.g. "CIN@NE" -- the second team listed is the home team.

    If `bold_team` is set, `team` (whichever side it's on) is wrapped in
    markdown bold, e.g. "CIN@**NE**" when `team` is the home side.
    """
    away, home = (opponent, team) if is_home else (team, opponent)
    if bold_team:
        if is_home:
            home = f"**{home}**"
        else:
            away = f"**{away}**"
    return f"{away}@{home}"


# Model-divergence coloring thresholds, based on the *absolute value* of
# the signed divergence (point-spread units, see
# models.win_prob.WinProbabilityResult.divergence -- positive means nfelo
# rates the home team more favorably than the market, negative means less
# favorably): |divergence| >= 3 is a meaningful market/Elo disagreement
# (red), 1 <= |divergence| < 3 is worth a second look (amber), otherwise
# unremarkable (default text color).
_DIVERGENCE_RED = "#cf222e"
_DIVERGENCE_AMBER = "#9a6700"


def _divergence_color_hex(divergence: Optional[float]) -> Optional[str]:
    """Hex color for a (signed) divergence value, or None for the default/unstyled case."""
    if divergence is None or (isinstance(divergence, float) and pd.isna(divergence)):
        return None
    magnitude = abs(divergence)
    if magnitude >= 3:
        return _DIVERGENCE_RED
    if magnitude >= 1:
        return _DIVERGENCE_AMBER
    return None


def _divergence_badge(divergence: Optional[float]) -> str:
    """Inline-styled HTML span showing a signed divergence value, colored per `_divergence_color_hex`.

    Returns "" when divergence is unavailable (no elo data for this game),
    so callers can safely splice this into an f-string unconditionally.
    Requires the containing st.markdown call to pass unsafe_allow_html=True.
    """
    if divergence is None or (isinstance(divergence, float) and pd.isna(divergence)):
        return ""
    color = _divergence_color_hex(divergence)
    style = f"color: {color}; font-weight: 600;" if color else ""
    return f' <span style="{style}">(model divergence {divergence:+.1f})</span>'


_TEAM_BIAS_COLOR = "#6639ba"


def _team_bias_badge(team_bias_adjustment: Optional[float], is_home: Optional[bool]) -> str:
    """Inline-styled HTML span showing a team's historical market-calibration bias.

    Returns "" whenever there's nothing to show (no adjustment applied, or
    an exact 0.0 -- functionally indistinguishable from "not computed" for
    display purposes), so callers can splice this into an f-string
    unconditionally. Requires the containing st.markdown call to pass
    unsafe_allow_html=True.
    """
    if team_bias_adjustment is None or (isinstance(team_bias_adjustment, float) and pd.isna(team_bias_adjustment)):
        return ""
    if team_bias_adjustment == 0.0:
        return ""
    context = "home" if is_home else "away"
    return (
        f' <span style="color: {_TEAM_BIAS_COLOR}; font-weight: 600;">'
        f"({team_bias_adjustment:+.1%} historical {context} bias)</span>"
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


@st.cache_data(show_spinner="Loading nfelo Elo ratings...")
def _get_elo_games() -> pd.DataFrame:
    return nfelo_client.load_nfelo_games()


@st.cache_data(show_spinner="Loading historical team performance...")
def _get_team_bias_games() -> pd.DataFrame:
    """Full historical nflverse games table, used for team_bias_adjustment (see models.win_prob)."""
    return nc.load_games()


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
    divergence: Optional[float] = None
    team_bias_adjustment: float = 0.0


def get_entry_recommendation(
    entry: str,
    season: int,
    week: int,
    used_teams: Set[str],
    schedule: pd.DataFrame,
    spread_model: wp.SpreadModel,
    exclude_teams: Set[str] = frozenset(),
    lookahead_weeks: int = dp_optimizer.DEFAULT_LOOKAHEAD_WEEKS,
    market_weight: float = 0.75,
    elo_games: Optional[pd.DataFrame] = None,
    team_bias_games: Optional[pd.DataFrame] = None,
) -> Optional[WeeklyRecommendation]:
    """Recommend a pick for `entry` ("A" or "B") using Entry A's DP-optimizer
    algorithm for both entries, excluding `exclude_teams` from the pool.

    `exclude_teams` is how one entry is kept off teams already shown for
    the other this week (e.g. Entry B is kept off whatever Entry A picked).
    `lookahead_weeks` / `market_weight` / `elo_games` / `team_bias_games`:
    see `models.win_prob.get_win_probability`.
    """
    raw_available = entry_a_value.build_candidates(
        season, week, used_teams, schedule=schedule, spread_model=spread_model,
        market_weight=market_weight, elo_games=elo_games, team_bias_games=team_bias_games,
    )
    available = [c for c in raw_available if c.team not in exclude_teams]
    if not available:
        return None
    try:
        rec = entry_a_value.recommend_pick(
            season,
            week,
            used_teams=used_teams,
            schedule=schedule,
            spread_model=spread_model,
            lookahead_weeks=lookahead_weeks,
            market_weight=market_weight,
            elo_games=elo_games,
            team_bias_games=team_bias_games,
        )
    except ValueError:
        rec = None
    projected_path = None
    if rec is not None and rec.team not in exclude_teams:
        top = next(c for c in available if c.team == rec.team)
        reasoning = rec.reasoning
        projected_path = rec.projected_path
    else:
        top = max(available, key=lambda c: c.win_probability)
        reasoning = (
            "The multi-week optimizer couldn't produce a valid plan this week "
            "(likely due to the other entry's exclusion); showing the single best option."
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
        divergence=top.divergence,
        team_bias_adjustment=top.team_bias_adjustment,
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


def _render_pick_line(pick: draft_order.DraftPick) -> None:
    spread_text = f", spread {pick.spread_line:+.1f}" if pick.spread_line is not None else ""
    matchup = _matchup_display(pick.team, pick.opponent, pick.is_home, bold_team=True)
    st.markdown(
        f"**Alternative:** {matchup} "
        f"— {pick.win_probability:.1%}{spread_text}{_divergence_badge(pick.divergence)}"
        f"{_team_bias_badge(pick.team_bias_adjustment, pick.is_home)}",
        unsafe_allow_html=True,
    )
    st.caption(pick.reasoning)


def _render_entry_column(
    label: str,
    entry: str,
    eliminated: bool,
    recommendation: Optional[WeeklyRecommendation],
    widget_key: str,
    extra_picks: Sequence[draft_order.DraftPick] = (),
) -> Optional[str]:
    """Render one entry's recommendation + pick dropdown; return the selected team (or None).

    `extra_picks` are additional alternate picks (from the draft order, see
    strategy.draft_order) shown below the primary recommendation -- e.g.
    Entry A's own pick #2, or every other candidate once the other entry
    has been eliminated.
    """
    st.markdown(f"### {label}")
    if eliminated:
        st.caption("Eliminated — no longer picking.")
        return None
    if recommendation is None:
        st.warning("No available team for this entry this week.")
        return None

    spread_text = f", spread {recommendation.spread_line:+.1f}" if recommendation.spread_line is not None else ""
    matchup = _matchup_display(recommendation.team, recommendation.opponent, recommendation.is_home, bold_team=True)
    st.markdown(
        f"**Recommends:** {matchup} — {recommendation.win_probability:.1%}{spread_text}"
        f"{_divergence_badge(recommendation.divergence)}"
        f"{_team_bias_badge(recommendation.team_bias_adjustment, recommendation.is_home)}",
        unsafe_allow_html=True,
    )
    st.caption(recommendation.reasoning)

    for pick in extra_picks:
        _render_pick_line(pick)

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
        if value == "LOSS":
            return "color: #cf222e; font-weight: 600;"
        return ""

    def _color_flag(value):
        return "color: #9a6700; font-weight: 600;" if value == "Override" else ""

    def _fmt_divergence(v):
        return f"{v:+.1f}" if isinstance(v, (int, float)) and pd.notna(v) else "—"

    def _color_divergence(value):
        if not isinstance(value, (int, float)) or pd.isna(value):
            return ""
        color = _divergence_color_hex(value)
        return f"color: {color}; font-weight: 600;" if color else ""

    def _fmt_team_bias(v):
        return f"{v:+.1%}" if isinstance(v, (int, float)) and pd.notna(v) and v != 0.0 else "—"

    def _color_team_bias(value):
        if not isinstance(value, (int, float)) or pd.isna(value) or value == 0.0:
            return ""
        return f"color: {_TEAM_BIAS_COLOR}; font-weight: 600;"

    result_cols = [c for c in df.columns if c.endswith("Result")]
    flag_cols = [c for c in df.columns if c.endswith("Match/Override")]
    win_prob_cols = [c for c in df.columns if c.endswith("Win Prob")]
    spread_cols = [c for c in df.columns if c.endswith("Spread")]
    divergence_cols = [c for c in df.columns if c.endswith("Model Divergence")]
    team_bias_cols = [c for c in df.columns if c.endswith("Team Bias")]

    format_map = {c: _fmt_pct for c in win_prob_cols}
    format_map.update({c: _fmt_spread for c in spread_cols})
    format_map.update({c: _fmt_divergence for c in divergence_cols})
    format_map.update({c: _fmt_team_bias for c in team_bias_cols})

    styler = df.style
    if result_cols:
        styler = styler.map(_color_result, subset=result_cols)
    if flag_cols:
        styler = styler.map(_color_flag, subset=flag_cols)
    if divergence_cols:
        styler = styler.map(_color_divergence, subset=divergence_cols)
    if team_bias_cols:
        styler = styler.map(_color_team_bias, subset=team_bias_cols)
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

    lookahead_options = list(range(1, 19))
    default_lookahead_index = lookahead_options.index(dp_optimizer.DEFAULT_LOOKAHEAD_WEEKS)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 2, 1])
    with col1:
        season = st.selectbox("Season", options=list(reversed(seasons)), index=0)
    with col2:
        starting_week = st.number_input("Starting Week", min_value=1, max_value=22, value=1, step=1)
    with col3:
        lookahead_weeks = st.selectbox(
            "Lookahead Weeks (N)",
            options=lookahead_options,
            index=default_lookahead_index,
            help="How many weeks ahead Entry A's DP optimizer plans over (see 'Prompt 3').",
        )
    with col4:
        weight_label = st.select_slider(
            "Market vs Elo Blend",
            options=list(MARKET_WEIGHT_OPTIONS.keys()),
            value=DEFAULT_MARKET_WEIGHT_LABEL,
            help=(
                "Blends market-derived win probability (moneylines/spread) with "
                "nfelo's Elo-model win probability. 100% Market matches the "
                "original behavior."
            ),
        )
        market_weight = MARKET_WEIGHT_OPTIONS[weight_label]
    with col5:
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
    elo_games = _get_elo_games() if market_weight < 1.0 else None
    team_bias_games = _get_team_bias_games()
    max_week = int(schedule["week"].max())

    st.divider()

    if eliminated_a and eliminated_b:
        st.error("Both entries have been eliminated. Click Reset Simulation to start over.")
    elif current_week > max_week:
        st.success(f"Reached the end of the {season} season!")
    else:
        st.subheader(f"Week {current_week}")

        # Draft-order alternates (see strategy.draft_order): normally Entry A's
        # own pick #2 and Entry B's own pick #4. Once one entry is eliminated,
        # the draft keeps drafting from the *surviving* entry's remaining
        # teams -- so its card shows every pick except the one duplicating
        # its own primary recommendation, still four distinct candidates
        # total, just no longer split across two cards.
        if eliminated_a and eliminated_b:
            draft = []
        else:
            if eliminated_a:
                draft_used_a, draft_used_b = used_b, used_b
            elif eliminated_b:
                draft_used_a, draft_used_b = used_a, used_a
            else:
                draft_used_a, draft_used_b = used_a, used_b
            try:
                draft = draft_order.draft_picks(
                    season,
                    current_week,
                    draft_used_a,
                    draft_used_b,
                    rounds=2,
                    schedule=schedule,
                    spread_model=spread_model,
                    lookahead_weeks=lookahead_weeks,
                    market_weight=market_weight,
                    elo_games=elo_games,
                    team_bias_games=team_bias_games,
                )
            except ValueError:
                draft = []

        if eliminated_a and not eliminated_b:
            extra_a, extra_b = [], [d for d in draft if d.pick_number != 3]
        elif eliminated_b and not eliminated_a:
            extra_a, extra_b = [d for d in draft if d.pick_number != 1], []
        else:
            extra_a = [d for d in draft if d.pick_number == 2]
            extra_b = [d for d in draft if d.pick_number == 4]

        rec_a = (
            None
            if eliminated_a
            else get_entry_recommendation(
                "A",
                season,
                current_week,
                used_a,
                schedule,
                spread_model,
                lookahead_weeks=lookahead_weeks,
                market_weight=market_weight,
                elo_games=elo_games,
                team_bias_games=team_bias_games,
            )
        )
        # Key includes lookahead_weeks and weight_label so the widget resets
        # to the new recommendation whenever either live control changes,
        # rather than Streamlit silently keeping the old selection (still a
        # valid option in the new list) as if the user had chosen it.
        pick_a_key = f"pick_a_{season}_{current_week}_{lookahead_weeks}_{weight_label}"
        col_a, col_b = st.columns(2)
        with col_a:
            selected_a = _render_entry_column(
                "Entry A", "A", eliminated_a, rec_a, pick_a_key, extra_picks=extra_a
            )

        # Entry B must never end up recommending (or offering) either of Entry
        # A's two displayed picks -- its own confirmed/selected pick and its
        # pick #2 alternate -- not just whichever one is currently selected.
        entry_a_pick2_team = next((d.team for d in draft if d.pick_number == 2), None)
        exclude_for_b = {selected_a} if selected_a else set()
        if entry_a_pick2_team:
            exclude_for_b.add(entry_a_pick2_team)
        rec_b = (
            None
            if eliminated_b
            else get_entry_recommendation(
                "B",
                season,
                current_week,
                used_b,
                schedule,
                spread_model,
                exclude_teams=exclude_for_b,
                market_weight=market_weight,
                elo_games=elo_games,
                team_bias_games=team_bias_games,
            )
        )
        with col_b:
            # Key includes lookahead_weeks/weight_label (same reasoning as
            # pick_a_key above) and selected_a, so Entry B's widget also
            # resets cleanly whenever Entry A's pick changes and its option
            # list shifts underneath it.
            selected_b = _render_entry_column(
                "Entry B",
                "B",
                eliminated_b,
                rec_b,
                f"pick_b_{season}_{current_week}_{lookahead_weeks}_{weight_label}_{selected_a}",
                extra_picks=extra_b,
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
                row = {"Week": current_week, "Blend": weight_label}

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
                            "A: Model Divergence": cand_a.divergence,
                            "A: Team Bias": cand_a.team_bias_adjustment,
                        }
                    )
                    if outcome_a == "LOSS":
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
                            "B: Model Divergence": cand_b.divergence,
                            "B: Team Bias": cand_b.team_bias_adjustment,
                        }
                    )
                    if outcome_b == "LOSS":
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
        st.caption(f"Current blend weighting: **{weight_label}** (see the 'Blend' column for each row's setting).")

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
