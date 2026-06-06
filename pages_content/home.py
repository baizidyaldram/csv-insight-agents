import streamlit as st
import pandas as pd
from utils.session import set_df, is_data_loaded, get_df


def render():
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 2.6rem; font-weight: 700; color: #e0e0f0; margin-bottom: 0.3rem;">
            🤖 CSV Insight Agents
        </h1>
        <p style="color: #64748b; font-size: 1.1rem; margin-top: 0;">
            Multi-agent AI system for automated data analysis, visualization, and insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Agent pipeline overview ───────────────────────────────────────────────
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
            ai_badge = '<span style="background:#1e1b4b;color:#818cf8;padding:2px 7px;border-radius:10px;font-size:0.7rem;margin-left:6px;">AI</span>' if uses_ai else '<span style="background:#064e3b;color:#34d399;padding:2px 7px;border-radius:10px;font-size:0.7rem;margin-left:6px;">Local</span>'
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
                st.success(f"✅ **{uploaded_file.name}** loaded successfully!")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

        st.markdown("**— or —**")

        if st.button("📦 Load Sample Dataset", use_container_width=True):
            try:
                df = pd.read_csv("data/sample_orders.csv")
                set_df(df, "sample_orders.csv")
                st.success("✅ Sample e-commerce orders dataset loaded!")
            except Exception as e:
                st.error(f"Could not load sample: {e}")

        # ── Data preview ──────────────────────────────────────────────────────
        if is_data_loaded():
            df = get_df()
            st.markdown("---")
            st.markdown("#### 👀 Data Preview")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", df.shape[1])
            c3.metric("Numeric Cols", df.select_dtypes(include="number").shape[1])
            c4.metric("Missing %", f"{(df.isnull().sum().sum() / df.size * 100):.1f}%")

            st.dataframe(df.head(10), use_container_width=True, height=280)

            st.markdown("---")
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #2a2a4e;border-radius:10px;padding:1rem;">
                <p style="color:#94a3b8;margin:0;font-size:0.88rem;">
                    ✅ Data loaded. Use the <strong style="color:#a78bfa;">sidebar</strong> 
                    to navigate through the agent pipeline — start with 
                    <strong style="color:#a78bfa;">🔍 Data Quality</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px dashed #2a2a4e;border-radius:10px;padding:1.2rem;margin-top:1rem;text-align:center;">
                <p style="color:#64748b;margin:0;font-size:0.88rem;">
                    Upload a CSV or load the sample dataset to unlock all agents.
                </p>
            </div>
            """, unsafe_allow_html=True)
