import streamlit as st
import pandas as pd
from utils.session import get_df, is_data_loaded


def render():
    st.markdown("## 📊 Statistical Analysis Agent")
    st.markdown("Descriptive statistics and correlations.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded.")
        if st.button("← Back to Home", use_container_width=False):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()
    
    if st.button("▶ Run Statistical Analysis"):
        st.session_state.stats_done = True
    
    if st.session_state.get("stats_done"):
        numeric_cols = df.select_dtypes(include="number").columns
        
        if len(numeric_cols) > 0:
            st.markdown("### Descriptive Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            if len(numeric_cols) >= 2:
                st.markdown("### Correlation Matrix")
                st.dataframe(df[numeric_cols].corr(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col2:
            if st.button("➡️ Next: Visualization", use_container_width=True):
                st.session_state.current_page = "visualization"
                st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col2:
            st.info("Click the button above to run statistical analysis.")
