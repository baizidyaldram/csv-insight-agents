import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.session import get_df, is_data_loaded

PLOTLY_THEME = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Vivid


def render():
    st.markdown("## 📈 Visualization Agent")
    st.markdown("Auto-detects your data shape and generates the most relevant charts — all interactive.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = _detect_date_cols(df)

    if st.button("▶ Generate Visualizations", use_container_width=False):
        st.session_state.viz_done = True

    if not st.session_state.get("viz_done"):
        st.info("Click **Generate Visualizations** to auto-build charts for your dataset.")
        return

    charts_rendered = 0

    # ── 1. Numeric distributions (histograms) ─────────────────────────────────
    if numeric_cols:
        st.markdown("### 📊 Numeric Distributions")
        n = min(len(numeric_cols), 6)
        cols_per_row = 2
        rows = (n + cols_per_row - 1) // cols_per_row

        for i in range(0, n, cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(numeric_cols[i:i+cols_per_row]):
                with row_cols[j]:
                    fig = px.histogram(
                        df, x=col, nbins=30,
                        title=f"Distribution of {col}",
                        template=PLOTLY_THEME,
                        color_discrete_sequence=["#a78bfa"],
                        marginal="box",
                    )
                    fig.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)
        charts_rendered += 1

    # ── 2. Correlation heatmap ─────────────────────────────────────────────────
    if len(numeric_cols) >= 3:
        st.markdown("### 🔗 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            title="Feature Correlation Matrix",
            template=PLOTLY_THEME,
        )
        fig.update_layout(height=450, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        charts_rendered += 1

    # ── 3. Categorical bar charts ─────────────────────────────────────────────
    if cat_cols:
        st.markdown("### 🏷️ Categorical Distributions")
        n_cat = min(len(cat_cols), 4)
        for i in range(0, n_cat, 2):
            row_cols = st.columns(2)
            for j, col in enumerate(cat_cols[i:i+2]):
                with row_cols[j]:
                    top_n = df[col].value_counts().head(12).reset_index()
                    top_n.columns = [col, "count"]
                    fig = px.bar(
                        top_n, x=col, y="count",
                        title=f"Top values in '{col}'",
                        template=PLOTLY_THEME,
                        color="count",
                        color_continuous_scale="Viridis",
                    )
                    fig.update_layout(height=320, margin=dict(t=40, b=60, l=20, r=20), showlegend=False)
                    fig.update_xaxes(tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)
        charts_rendered += 1

    # ── 4. Time series (if date column detected) ───────────────────────────────
    if date_cols and numeric_cols:
        st.markdown("### 📅 Time Series")
        date_col = date_cols[0]
        y_col = st.selectbox("Select metric to plot over time", numeric_cols, key="ts_col")
        try:
            df_ts = df.copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors="coerce")
            df_ts = df_ts.dropna(subset=[date_col, y_col])
            df_ts = df_ts.sort_values(date_col)

            # Group by month if many dates
            if df_ts[date_col].nunique() > 60:
                df_ts["period"] = df_ts[date_col].dt.to_period("M").dt.to_timestamp()
                grouped = df_ts.groupby("period")[y_col].sum().reset_index()
                grouped.columns = [date_col, y_col]
                df_ts = grouped

            fig = px.line(
                df_ts, x=date_col, y=y_col,
                title=f"{y_col} over time",
                template=PLOTLY_THEME,
                color_discrete_sequence=["#34d399"],
                markers=True,
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1
        except Exception as e:
            st.warning(f"Could not render time series: {e}")

    # ── 5. Scatter plot: top 2 numeric cols ────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.markdown("### 🔵 Scatter Plot")
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
        y_col = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
        color_col = c3.selectbox("Color by", ["None"] + cat_cols, index=0, key="scatter_c")

        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_col if color_col != "None" else None,
            title=f"{x_col} vs {y_col}",
            template=PLOTLY_THEME,
            color_discrete_sequence=COLOR_SEQ,
            opacity=0.7,
            hover_data=df.columns[:5].tolist(),
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        charts_rendered += 1

    # ── 6. Box plots for numeric cols ─────────────────────────────────────────
    if numeric_cols and cat_cols:
        st.markdown("### 📦 Box Plots by Category")
        num_col = st.selectbox("Numeric column", numeric_cols, key="box_num")
        cat_col = st.selectbox("Group by", cat_cols, key="box_cat")
        top_cats = df[cat_col].value_counts().head(10).index
        df_box = df[df[cat_col].isin(top_cats)]
        fig = px.box(
            df_box, x=cat_col, y=num_col,
            title=f"{num_col} by {cat_col}",
            template=PLOTLY_THEME,
            color=cat_col,
            color_discrete_sequence=COLOR_SEQ,
        )
        fig.update_layout(height=420, showlegend=False)
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        charts_rendered += 1

    if charts_rendered == 0:
        st.warning("Not enough data variety to generate charts. Ensure your CSV has numeric or categorical columns.")

    st.markdown("---")
    st.markdown("""
    <div style="background:#1e1e2e;border:1px solid #2a2a4e;border-radius:10px;padding:1rem;">
        <p style="color:#94a3b8;margin:0;font-size:0.88rem;">
            ➡️ Next step: head to <strong style="color:#a78bfa;">💡 AI Insights</strong> for LLM-powered analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)


def _detect_date_cols(df: pd.DataFrame) -> list:
    date_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(20)
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() / len(sample) > 0.7:
                    date_cols.append(col)
            except Exception:
                pass
        elif "datetime" in str(df[col].dtype):
            date_cols.append(col)
    return date_cols
