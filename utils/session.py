import streamlit as st
import pandas as pd


def init_session():
    """Initialize all session state keys."""
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "file_name": None,
        "quality_report": None,
        "cleaning_report": None,
        "stats_report": None,
        "insights_text": None,
        "report_text": None,
        "current_page": "home",
        "stats_done": False,
        "viz_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def set_df(df: pd.DataFrame, file_name: str = "data.csv"):
    """Store uploaded dataframe."""
    st.session_state.df_raw = df
    st.session_state.df_clean = df.copy()
    st.session_state.file_name = file_name
    # Clear downstream results
    st.session_state.quality_report = None
    st.session_state.cleaning_report = None
    st.session_state.stats_report = None
    st.session_state.insights_text = None
    st.session_state.report_text = None
    st.session_state.stats_done = False
    st.session_state.viz_done = False


def get_df(prefer_clean: bool = True) -> pd.DataFrame | None:
    """Return the active dataframe."""
    if prefer_clean and st.session_state.df_clean is not None:
        return st.session_state.df_clean
    return st.session_state.df_raw


def get_raw_df() -> pd.DataFrame | None:
    return st.session_state.df_raw


def is_data_loaded() -> bool:
    return st.session_state.df_raw is not None


def update_clean_df(df: pd.DataFrame):
    st.session_state.df_clean = df
