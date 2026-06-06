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

# ── CSS for Perfect Centered Layout ──────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Main App Background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
}

/* Main content - PERFECTLY CENTERED */
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1000px !important;
    margin: 0 auto !important;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border-radius: 28px;
    margin-top: 2rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Hide Streamlit Branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar - COLLAPSIBLE with proper styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 12, 41, 0.98) 0%, rgba(36, 36, 62, 0.98) 100%);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.15);
    transition: all 0.3s ease;
}

[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 280px;
    width: 280px;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0px;
    width: 0px;
    overflow: hidden;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar toggle button - VISIBLE AND CLICKABLE */
[data-testid="collapsedControl"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 0 12px 12px 0;
    padding: 10px 14px;
    cursor: pointer;
    transition: all 0.3s ease;
    z-index: 999999;
    position: fixed;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
}

[data-testid="collapsedControl"]:hover {
    background: linear-gradient(135deg, #764ba2, #667eea);
    transform: translateY(-50%) scale(1.05);
}

/* Sidebar Content Padding */
[data-testid="stSidebar"] .element-container {
    padding: 0 0.75rem;
}

/* Sidebar Buttons */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    text-align: left;
    padding: 0.7rem 1rem;
    transition: all 0.3s ease;
    font-weight: 500;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateX(5px);
    border-color: rgba(255, 255, 255, 0.3);
}

/* Primary Button in Sidebar */
[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

/* Cards */
.agent-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.05));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}

.agent-card:hover {
    transform: translateY(-3px);
    border-color: rgba(255, 255, 255, 0.3);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.08));
}

.agent-card h4 {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    background: linear-gradient(135deg, #fff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.agent-card p {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.7);
    margin: 0;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.08));
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.7);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
}

/* Headers */
h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem !important;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #ffffff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: rgba(255, 255, 255, 0.9) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Badges */
.badge-ready {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
}

.badge-waiting {
    background: linear-gradient(135deg, #f2994a, #f2c94c);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
}

/* Divider */
hr {
    margin: 1.5rem 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
}

/* Center text utility */
.text-center {
    text-align: center;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.05);
    border: 2px dashed rgba(255, 255, 255, 0.3);
    border-radius: 20px;
    padding: 1rem;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
init_session()

# ── Sidebar Navigation (Collapsible) ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CSV Insight Agents")
    st.markdown("### AI-Powered Analysis")
    st.markdown("---")
    
    # Navigation buttons
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
        
        button_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{key}", use_container_width=True, disabled=disabled, type=button_type):
            st.session_state.current_page = key
            st.rerun()
    
    st.markdown("---")
    
    # Data Status
    if is_data_loaded():
        df = get_df()
        st.markdown(f'<span class="badge-ready">✓ Data Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"### {df.shape[0]:,}")
        st.caption("rows")
        st.markdown(f"### {df.shape[1]}")
        st.caption("columns")
        
        with st.expander("📊 Quick Preview"):
            st.dataframe(df.head(3), use_container_width=True)
    else:
        st.markdown(f'<span class="badge-waiting">⏳ No Data</span>', unsafe_allow_html=True)
        st.caption("Upload a CSV to begin")
    
    st.markdown("---")
    st.caption("💡 Click the ◀ arrow to collapse")

# ── Page Router ───────────────────────────────────────────────────────────────
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
    st.error(f"Page not found: {e}")
    st.info("Please ensure all page files exist in the 'pages_content' folder")
