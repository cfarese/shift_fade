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
## pages
## ---------------------------------------------------------------------------

if page == "Overview":
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
