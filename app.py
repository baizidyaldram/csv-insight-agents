import streamlit as st
from utils.session import init_session, get_df, is_data_loaded
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Insight Agents | AI-Powered Data Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Enhanced Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

/* Main container */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar Styles - FIXED COLLAPSE ISSUE */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 12, 41, 0.95) 0%, rgba(36, 36, 62, 0.95) 100%);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(108, 99, 255, 0.3);
    transition: all 0.3s ease;
}

[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 280px;
    width: 280px;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: -280px;
}

/* Sidebar content */
[data-testid="stSidebar"] * {
    color: #f0f0ff !important;
}

/* Custom sidebar toggle button */
[data-testid="collapsedControl"] {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    border-radius: 0 12px 12px 0;
    padding: 8px 12px;
    cursor: pointer;
    transition: all 0.3s ease;
}

[data-testid="collapsedControl"]:hover {
    background: linear-gradient(135deg, #7c74ff, #b89bff);
    transform: scale(1.05);
}

/* Main content area */
.main .block-container {
    background: transparent;
    padding: 2rem;
}

/* Glass morphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 20px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(108, 99, 255, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(108, 99, 255, 0.2);
}

/* Agent cards */
.agent-card {
    background: linear-gradient(135deg, rgba(30, 30, 46, 0.8), rgba(22, 33, 62, 0.8));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: all 0.3s ease;
    cursor: pointer;
}
.agent-card:hover {
    border-color: #6c63ff;
    transform: translateX(5px);
    box-shadow: 0 5px 20px rgba(108, 99, 255, 0.3);
}
.agent-card h4 {
    background: linear-gradient(135deg, #a78bfa, #6c63ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
    font-size: 0.95rem;
    font-weight: 700;
}
.agent-card p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.82rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(167, 139, 250, 0.1));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #6c63ff;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #6c63ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(30, 30, 46, 0.6);
    backdrop-filter: blur(10px);
    border: 2px dashed rgba(108, 99, 255, 0.4);
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6c63ff;
    background: rgba(108, 99, 255, 0.1);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.8rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4);
    opacity: 0.95;
}

/* Status badges */
.badge-ready {
    background: linear-gradient(135deg, #064e3b, #065f46);
    color: #34d399;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}
.badge-waiting {
    background: linear-gradient(135deg, #1e1b4b, #2e2a5e);
    color: #818cf8;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6c63ff, #a78bfa);
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    background: rgba(30, 30, 46, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 0.5rem;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: rgba(30, 30, 46, 0.6);
    padding: 0.5rem;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(108, 99, 255, 0.2);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white;
}

/* Headers */
h1, h2, h3 {
    background: linear-gradient(135deg, #e0e0f0, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #6c63ff, #a78bfa, transparent);
    margin: 2rem 0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(30, 30, 46, 0.6);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
init_session()

# ── Sidebar navigation with FIXED COLLAPSE ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CSV Insight Agents")
    st.markdown("### AI-Powered Data Analysis")
    st.markdown("---")

    pages = {
        "🏠 Home": "home",
        "🔍 Data Quality": "quality",
        "🧹 Data Cleaning": "cleaning",
        "📊 Statistical Analysis": "stats",
        "📈 Visualization": "visualization",
        "💡 AI Insights": "insights",
        "📄 Report": "report",
    }

    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    for label, key in pages.items():
        is_active = st.session_state.current_page == key
        disabled = key != "home" and not is_data_loaded()
        
        # Custom button styling with active state
        button_style = "primary" if is_active else "secondary"
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            disabled=disabled,
            type=button_style,
        ):
            st.session_state.current_page = key
            st.rerun()

    st.markdown("---")
    
    # Data status card
    if is_data_loaded():
        df = get_df()
        st.markdown(f'<span class="badge-ready">✓ Data Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"### {df.shape[0]:,}")
        st.caption("rows")
        st.markdown(f"### {df.shape[1]}")
        st.caption("columns")
        
        # Mini preview
        with st.expander("📊 Quick Preview"):
            st.dataframe(df.head(5), use_container_width=True)
    else:
        st.markdown('<span class="badge-waiting">⏳ No data yet</span>', unsafe_allow_html=True)
        st.caption("Upload a CSV on the Home page to begin.")
        st.info("💡 **Tip**: Try the sample dataset!")

# ── Page router ───────────────────────────────────────────────────────────────
page = st.session_state.current_page

# Import pages dynamically
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
