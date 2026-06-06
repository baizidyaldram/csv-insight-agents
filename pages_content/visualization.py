import streamlit as st
import pandas as pd
import plotly.express as px
from utils.session import get_df, is_data_loaded


def render():
    st.markdown("## 📈 Visualization Agent")
    st.markdown("Auto-detects your data shape and generates the most relevant charts — all interactive.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()
    
    # Remove any index column if it's unnamed or looks like an index
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'index' in df.columns and df['index'].equals(pd.RangeIndex(start=0, stop=len(df))):
        df = df.drop(columns=['index'])
    
    if st.button("▶ Generate Visualizations", use_container_width=False):
        st.session_state.viz_done = True
    
    if st.session_state.get("viz_done"):
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        
        # Remove any index-like columns from numeric_cols
        numeric_cols = [col for col in numeric_cols if col not in ['index', 'Unnamed: 0', 'level_0']]
        
        charts_rendered = 0
        
        # Numeric distributions
        if numeric_cols:
            st.markdown("### 📊 Numeric Distributions")
            for i, col in enumerate(numeric_cols[:4]):
                fig = px.histogram(
                    df, x=col, nbins=30,
                    title=f"Distribution of {col}",
                    template="plotly_dark",
                    color_discrete_sequence=["#a78bfa"],
                )
                fig.update_layout(height=400, margin=dict(t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1
        
        # Correlation heatmap
        if len(numeric_cols) >= 2:
            st.markdown("### 🔗 Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1,
                title="Feature Correlation Matrix",
                template="plotly_dark",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1
        
        # Categorical bar charts
        if cat_cols:
            st.markdown("### 🏷️ Categorical Distributions")
            for col in cat_cols[:3]:
                top_n = df[col].value_counts().head(10).reset_index()
                top_n.columns = [col, "count"]
                fig = px.bar(
                    top_n, x=col, y="count",
                    title=f"Top values in '{col}'",
                    template="plotly_dark",
                    color="count",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(height=400)
                fig.update_xaxes(tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1
        
        if charts_rendered == 0:
            st.warning("Not enough data variety to generate charts.")
        
        # Next button
        if st.button("➡️ Next: AI Insights", use_container_width=False):
            st.session_state.current_page = "insights"
            st.rerun()
    else:
        st.info("Click **Generate Visualizations** to auto-build charts for your dataset.")
