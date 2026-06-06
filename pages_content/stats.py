import streamlit as st
import pandas as pd
from utils.session import get_df, is_data_loaded
from utils.data_profiler import generate_data_profile, display_data_profile


def render():
    st.markdown("## 📊 Statistical Analysis Agent")
    st.markdown("Descriptive statistics, correlations, and advanced data profiling.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded.")
        if st.button("← Back to Home", use_container_width=False):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()
    
    # Add Data Profiling button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("▶ Run Statistical Analysis", use_container_width=False):
            st.session_state.stats_done = True
        if st.button("📊 Generate Data Profile", use_container_width=False):
            with st.spinner("Generating comprehensive data profile..."):
                profile = generate_data_profile(df)
                st.session_state.data_profile = profile
                st.session_state.stats_done = True
    
    if st.session_state.get("stats_done"):
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📐 Descriptive Stats", "🔗 Correlations", "📊 Data Profile", "🏷️ Categorical"])
        
        # Tab 1: Descriptive Statistics
        with tab1:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                st.markdown("### Descriptive Statistics")
                desc = df[numeric_cols].describe()
                
                # Add additional stats
                desc.loc['skewness'] = df[numeric_cols].skew()
                desc.loc['kurtosis'] = df[numeric_cols].kurtosis()
                desc.loc['missing'] = df[numeric_cols].isnull().sum()
                desc.loc['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
                
                st.dataframe(desc.round(3), use_container_width=True)
            else:
                st.info("No numeric columns found in this dataset.")
        
        # Tab 2: Correlations
        with tab2:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) >= 2:
                st.markdown("### Correlation Matrix")
                corr = df[numeric_cols].corr()
                st.dataframe(corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.3f}"), use_container_width=True)
                
                # Top correlations
                st.markdown("### Top Correlated Pairs")
                corr_pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        corr_pairs.append({
                            "Variable A": corr.columns[i],
                            "Variable B": corr.columns[j],
                            "Correlation": corr.iloc[i, j],
                            "Strength": "Strong" if abs(corr.iloc[i, j]) > 0.7 else "Moderate" if abs(corr.iloc[i, j]) > 0.4 else "Weak"
                        })
                corr_df = pd.DataFrame(corr_pairs).sort_values("Correlation", key=abs, ascending=False)
                st.dataframe(corr_df.head(10), use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
        
        # Tab 3: Data Profile (NEW)
        with tab3:
            if st.session_state.get("data_profile"):
                display_data_profile(st.session_state.data_profile)
            else:
                st.info("Click 'Generate Data Profile' above to see comprehensive data analysis.")
                if st.button("Generate Data Profile Now"):
                    with st.spinner("Generating profile..."):
                        profile = generate_data_profile(df)
                        st.session_state.data_profile = profile
                        st.rerun()
        
        # Tab 4: Categorical
        with tab4:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols) > 0:
                selected_cat = st.selectbox("Select categorical column", cat_cols)
                if selected_cat:
                    value_counts = df[selected_cat].value_counts()
                    st.markdown(f"### Distribution of {selected_cat}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Unique Values", df[selected_cat].nunique())
                    with col2:
                        st.metric("Most Common", str(value_counts.index[0]) if len(value_counts) > 0 else "N/A")
                    with col3:
                        st.metric("Missing", df[selected_cat].isnull().sum())
                    
                    # Show value counts
                    freq_df = value_counts.reset_index()
                    freq_df.columns = ["Value", "Count"]
                    freq_df["Percentage"] = (freq_df["Count"] / len(df) * 100).round(2)
                    st.dataframe(freq_df.head(20), use_container_width=True)
            else:
                st.info("No categorical columns found in this dataset.")
        
        # Navigation buttons
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("← Back to Cleaning", use_container_width=True):
                st.session_state.current_page = "cleaning"
                st.rerun()
        with col_nav2:
            if st.button("➡️ Next: Visualization", use_container_width=True):
                st.session_state.current_page = "visualization"
                st.rerun()
    else:
        st.info("Click **Run Statistical Analysis** or **Generate Data Profile** to begin.")
