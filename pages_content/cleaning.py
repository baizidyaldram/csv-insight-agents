import streamlit as st
import pandas as pd
from utils.session import get_df, get_raw_df, update_clean_df, is_data_loaded


def render():
    st.markdown("## 🧹 Data Cleaning Agent")
    st.markdown("Automatically detects and fixes common data issues.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded.")
        return

    df_raw = get_raw_df()
    
    if st.button("▶ Run Data Cleaning"):
        df_clean = df_raw.copy()
        
        # Remove duplicates
        before = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        removed = before - len(df_clean)
        
        # Fill missing numeric with median
        numeric_cols = df_clean.select_dtypes(include="number").columns
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        update_clean_df(df_clean)
        st.session_state.cleaning_report = {
            "rows_removed": removed,
            "before_shape": df_raw.shape,
            "after_shape": df_clean.shape,
        }
        st.success("✅ Cleaning complete!")
    
    if st.session_state.get("cleaning_report"):
        r = st.session_state.cleaning_report
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows Before", f"{r['before_shape'][0]:,}")
        col2.metric("Rows After", f"{r['after_shape'][0]:,}")
        col3.metric("Rows Removed", f"{r['rows_removed']:,}")
        
        st.dataframe(get_df().head(10), use_container_width=True)
        
        if st.button("➡️ Next: Statistical Analysis"):
            st.session_state.current_page = "stats"
            st.rerun()
    else:
        st.info("Click the button above to clean your data.")
