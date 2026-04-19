## Main Streamlit dashboard entry point.
##
## Run with: streamlit run dashboard/app.py
##
## Pages:
##   Overview    -- league-wide decay coefficient distribution
##   Players     -- per-player RAPM + decay curve
##   Lines       -- line combination stats and overuse flags
##   Team Report -- full team breakdown with overuse alerts

import sys
from pathlib import Path

## make sure imports resolve from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.settings import cfg

st.set_page_config(
    page_title="Hockey Shift Decay RAPM",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #f8f8f7 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e8e8 !important;
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Sans', sans-serif !important; }
h1, h2, h3, h4 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    letter-spacing: -0.015em !important;
    font-weight: 700 !important;
    color: #111 !important;
}
/* KPI metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 6px;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #888 !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #111 !important;
}
/* Tabs */
[data-testid="stTab"] button {
    font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    letter-spacing: 0.01em !important;
}
/* Dataframe border */
.stDataFrame { border-radius: 6px !important; overflow: hidden; }
/* Caption italic */
[data-testid="stCaptionContainer"] p {
    font-style: italic !important;
    color: #888 !important;
    font-size: 12px !important;
}
/* Subheader weight */
[data-testid="stHeadingWithActionElements"] h2 {
    font-size: 16px !important;
    letter-spacing: -0.01em !important;
}
/* Selectbox font */
[data-baseweb="select"] { font-family: 'IBM Plex Sans', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

## ---------------------------------------------------------------------------
## sidebar nav
## ---------------------------------------------------------------------------

PAGES = ["Overview", "Players", "Lines", "Team Report"]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

SEASONS = cfg.seasons
season = st.sidebar.selectbox("Season", SEASONS, index=len(SEASONS) - 1)

st.sidebar.markdown("---")
st.sidebar.caption("Shift Decay RAPM v0.1")

## ---------------------------------------------------------------------------
## data loading with caching so re-renders don't re-read parquet
## ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_rapm(season: str) -> pd.DataFrame | None:
    path = cfg.paths.processed / f"rapm_results_{season}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_stints(season: str) -> pd.DataFrame | None:
    path = cfg.paths.processed / f"stints_{season}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _no_data_warning(what: str):
    st.warning(f"No {what} data found for {season}. Run the pipeline first.")
    st.code(f"python -m src.ingestion.pipeline --season {season}\n"
            f"python -m src.features.export_matrix --season {season}\n"
            f"Rscript r/rapm/rapm_model.R --season {season}")


## ---------------------------------------------------------------------------
## player profile routing
##
## views call st.rerun() after setting selected_player_id.
## on that rerun, record _player_page (the sidebar page when selection happened)
## so we can detect when the user navigates away via the sidebar.
## ---------------------------------------------------------------------------

pid = st.session_state.get("selected_player_id")

if pid is not None and "_player_page" not in st.session_state:
    ## first rerun after a player was selected -- lock in the source page
    st.session_state["_player_page"] = page

if pid is not None and st.session_state.get("_player_page") != page:
    ## user clicked a different sidebar page -- clear the profile and rerun
    st.session_state["selected_player_id"] = None
    st.session_state.pop("_player_page", None)
    st.rerun()

## ---------------------------------------------------------------------------
## pages -- player profile takes over the whole screen when a player is selected
## ---------------------------------------------------------------------------

if st.session_state.get("selected_player_id") is not None:
    from dashboard._views.player_profile import render as render_profile
    render_profile(st.session_state["selected_player_id"], season, _load_rapm(season))

elif page == "Overview":
    from dashboard._views.overview import render
    render(season, _load_rapm(season))

elif page == "Players":
    from dashboard._views.players import render
    render(season, _load_rapm(season))

elif page == "Lines":
    from dashboard._views.lines import render
    render(season, _load_stints(season))

elif page == "Team Report":
    from dashboard._views.team_report import render
    render(season, _load_rapm(season), _load_stints(season))
