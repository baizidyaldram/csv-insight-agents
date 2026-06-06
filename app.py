import streamlit as st
from utils.session import init_session, get_df, is_data_loaded
import pandas as pd
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Insight Agents | AI-Powered Data Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern CSS with Beautiful Background ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

/* Beautiful Animated Background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
}

/* Main content area */
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border-radius: 32px;
    margin-top: 1rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 25px 45px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Cards */
.glass-card, .agent-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 24px;
    padding: 1.5rem;
    transition: all 0.4s ease;
}

.agent-card:hover {
    transform: translateY(-5px);
    border-color: rgba(255, 255, 255, 0.4);
}

.agent-card h4 {
    background: linear-gradient(135deg, #fff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 50px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #764ba2, #667eea);
}

/* Headers */
h1, h2, h3 {
    background: linear-gradient(135deg, #ffffff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Badges */
.badge-ready {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

.badge-waiting {
    background: linear-gradient(135deg, #f2994a, #f2c94c);
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    padding: 0.5rem;
    border-radius: 60px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    color: rgba(255, 255, 255, 0.7);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
init_session()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CSV Insight Agents")
    st.markdown("### AI-Powered Analysis")
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
        st.markdown(f"### {df.shape[0]:,}")
        st.caption("rows")
        st.markdown(f"### {df.shape[1]}")
        st.caption("columns")
    else:
        st.markdown('<span class="badge-waiting">⏳ No data yet</span>', unsafe_allow_html=True)
        st.caption("Upload a CSV on the Home page")

# ── Page router ───────────────────────────────────────────────────────────────
page = st.session_state.current_page

try:
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
except ImportError as e:
    st.error(f"Error loading page: {e}")
    st.info("Please make sure all page files exist in the 'pages_content' folder")
