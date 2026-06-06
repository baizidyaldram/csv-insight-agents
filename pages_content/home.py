import streamlit as st
import pandas as pd
from utils.session import set_df, is_data_loaded, get_df


def render():
    # ── Hero Section with animation ──────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem;">
            🤖 CSV Insight Agents
        </h1>
        <p style="font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem;">
            Multi-agent AI system for automated data analysis, visualization, and insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Agent pipeline overview with better visuals ───────────────────────────
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 🔄 Agent Pipeline")
        agents = [
            ("🔍", "Data Quality Agent", "Scores completeness, validity, consistency", False),
            ("🧹", "Data Cleaning Agent", "Fixes missing values, duplicates, type issues", False),
            ("📊", "Statistical Analysis Agent", "Descriptive stats, correlations, distributions", False),
            ("📈", "Visualization Agent", "Auto-generates relevant charts with Plotly", False),
            ("💡", "AI Insights Agent", "LLM-powered narrative business insights", True),
            ("📄", "Report Writing Agent", "Compiles full analysis into downloadable report", True),
        ]
        
        for icon, name, desc, uses_ai in agents:
            ai_badge = '<span style="background:linear-gradient(135deg,#1e1b4b,#2e2a5e);color:#818cf8;padding:2px 8px;border-radius:12px;font-size:0.7rem;margin-left:8px;">AI</span>' if uses_ai else '<span style="background:linear-gradient(135deg,#064e3b,#065f46);color:#34d399;padding:2px 8px;border-radius:12px;font-size:0.7rem;margin-left:8px;">Local</span>'
            st.markdown(f"""
            <div class="agent-card">
                <h4>{icon} {name} {ai_badge}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📁 Upload Your CSV")
        
        uploaded_file = st.file_uploader(
            "Drop a CSV file here",
            type=["csv"],
            help="Any CSV file works — sales data, survey results, financial records, etc.",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                set_df(df, uploaded_file.name)
                st.balloons()
                st.success(f"✅ **{uploaded_file.name}** loaded successfully!")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

        st.markdown("**— or —**")

        if st.button("📦 Load Sample Dataset", use_container_width=True):
            try:
                df = pd.read_csv("data/sample_orders.csv")
                set_df(df, "sample_orders.csv")
                st.success("✅ Sample e-commerce orders dataset loaded!")
                st.balloons()
            except Exception as e:
                st.error(f"Could not load sample: {e}")

        # ── Data preview with enhanced metrics ──────────────────────────────────────
        if is_data_loaded():
            df = get_df()
            st.markdown("---")
            st.markdown("#### 👀 Data Preview")

            # Enhanced metric cards
            col_metrics = st.columns(4)
            metrics_data = [
                ("Rows", f"{df.shape[0]:,}", "📊"),
                ("Columns", df.shape[1], "📋"),
                ("Numeric Cols", df.select_dtypes(include="number").shape[1], "🔢"),
                ("Missing %", f"{(df.isnull().sum().sum() / df.size * 100):.1f}%", "⚠️"),
            ]
            
            for col, (label, value, icon) in zip(col_metrics, metrics_data):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 2rem;">{icon}</div>
                        <div class="value">{value}</div>
                        <div class="label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.dataframe(df.head(10), use_container_width=True, height=280)

            # Next steps button
            if st.button("🚀 Start Analysis →", use_container_width=True):
                st.session_state.current_page = "quality"
                st.rerun()
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; margin-top: 1rem;">
                <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">
                    📂 Upload a CSV or load the sample dataset to unlock all agents.
                </p>
            </div>
            """, unsafe_allow_html=True)
