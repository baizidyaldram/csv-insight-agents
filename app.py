import streamlit as st
from utils.session import init_session, get_df, is_data_loaded
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Insight Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a3e;
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

/* Cards */
.agent-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #16213e 100%);
    border: 1px solid #2a2a4e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.agent-card:hover { border-color: #6c63ff; }
.agent-card h4 { color: #a78bfa; margin: 0 0 0.3rem 0; font-size: 0.95rem; }
.agent-card p  { color: #94a3b8; margin: 0; font-size: 0.82rem; }

/* Metric cards */
.metric-box {
    background: #1e1e2e;
    border: 1px solid #2a2a4e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-box .value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-box .label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #1e1e2e;
    border: 2px dashed #2a2a4e;
    border-radius: 12px;
    padding: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Status badge */
.badge-ready   { background: #064e3b; color: #34d399; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-waiting { background: #1e1b4b; color: #818cf8; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

/* Divider */
hr { border-color: #2a2a4e; }

/* Code font */
code, pre { font-family: 'JetBrains Mono', monospace; }

/* Main background */
.main .block-container { background: #0d0d1a; }
</style>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
init_session()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CSV Insight Agents")
    st.markdown("---")

    pages = {
        "🏠  Home": "home",
        "🔍  Data Quality": "quality",
        "🧹  Data Cleaning": "cleaning",
        "📊  Statistical Analysis": "stats",
        "📈  Visualization": "visualization",
        "💡  AI Insights": "insights",
        "📄  Report": "report",
    }

    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    for label, key in pages.items():
        is_active = st.session_state.current_page == key
        disabled = key != "home" and not is_data_loaded()
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            disabled=disabled,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = key
            st.rerun()

    st.markdown("---")
    if is_data_loaded():
        df = get_df()
        st.markdown(f'<span class="badge-ready">✓ Data Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"**{df.shape[0]:,}** rows · **{df.shape[1]}** cols")
    else:
        st.markdown('<span class="badge-waiting">⏳ No data yet</span>', unsafe_allow_html=True)
        st.caption("Upload a CSV on the Home page to begin.")

# ── Page router ───────────────────────────────────────────────────────────────
page = st.session_state.current_page

if page == "home":
    from pages_content.home import render
    render()
elif page == "quality":
    from pages_content.quality import render
    render()
elif page == "cleaning":
    from pages_content.cleaning import render
    render()
elif page == "stats":
    from pages_content.stats import render
    render()
elif page == "visualization":
    from pages_content.visualization import render
    render()
elif page == "insights":
    from pages_content.insights import render
    render()
elif page == "report":
    from pages_content.report import render
    render()
