import streamlit as st
import pandas as pd


def init_session():
    """Initialize all session state keys on first load."""
    defaults = {
        "df_raw": None,          # Original uploaded dataframe
        "df_clean": None,        # Cleaned dataframe (after cleaning agent)
        "file_name": None,       # Uploaded file name
        "quality_report": None,  # Output from Data Quality Agent
        "cleaning_report": None, # Output from Data Cleaning Agent
        "stats_report": None,    # Output from Statistical Analysis Agent
        "insights_text": None,   # Output from AI Insights Agent
        "report_text": None,     # Output from Report Writing Agent
        "current_page": "home",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def set_df(df: pd.DataFrame, file_name: str = "data.csv"):
    """Store uploaded dataframe into session."""
    st.session_state.df_raw = df
    st.session_state.df_clean = df.copy()
    st.session_state.file_name = file_name
    # Clear downstream results when new data is loaded
    st.session_state.quality_report = None
    st.session_state.cleaning_report = None
    st.session_state.stats_report = None
    st.session_state.insights_text = None
    st.session_state.report_text = None


def get_df(prefer_clean: bool = True) -> pd.DataFrame | None:
    """Return the active dataframe (clean if available, else raw)."""
    if prefer_clean and st.session_state.df_clean is not None:
        return st.session_state.df_clean
    return st.session_state.df_raw


def get_raw_df() -> pd.DataFrame | None:
    return st.session_state.df_raw


def is_data_loaded() -> bool:
    return st.session_state.df_raw is not None


def update_clean_df(df: pd.DataFrame):
    """Called by cleaning agent after it processes the data."""
    st.session_state.df_clean = df
