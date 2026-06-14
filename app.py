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

# ── Premium Dark Theme CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* Global Reset */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Premium Dark Gradient Background */
.stApp {
    background: radial-gradient(ellipse at 20% 30%, #1a1a2e, #0a0a15);
    background-attachment: fixed;
}

/* Animated particles effect */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(100, 108, 255, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 90% 60%, rgba(167, 139, 250, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 50% 80%, rgba(240, 147, 251, 0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* Main content container */
.main .block-container {
    padding: 2rem 2rem !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: rgba(18, 18, 35, 0.7);
    backdrop-filter: blur(20px);
    border-radius: 32px;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

/* Hide Streamlit Branding */
#MainMenu, footer, header { visibility: hidden; }

/* ============ SIDEBAR STYLING ============ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0a0a12 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] * {
    color: #e2e2f0 !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    text-align: left;
    padding: 0.7rem 1rem;
    transition: all 0.25s ease;
    font-weight: 500;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(100, 108, 255, 0.15);
    border-color: rgba(100, 108, 255, 0.4);
    transform: translateX(4px);
}

[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #6468ff, #a78bfa);
    border: none;
    box-shadow: 0 4px 12px rgba(100, 108, 255, 0.3);
}

/* ============ CARD STYLES ============ */
.agent-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.agent-card:hover {
    transform: translateY(-3px);
    border-color: rgba(100, 108, 255, 0.4);
    background: linear-gradient(135deg, rgba(100, 108, 255, 0.08), rgba(167, 139, 250, 0.04));
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
}

.agent-card h4 {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    background: linear-gradient(135deg, #ffffff, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.agent-card p {
    font-size: 0.8rem;
    color: #a0a0c0;
    margin: 0;
}

/* ============ METRIC CARDS ============ */
.metric-card {
    background: linear-gradient(135deg, rgba(100, 108, 255, 0.1), rgba(167, 139, 250, 0.05));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(100, 108, 255, 0.2);
    border-radius: 20px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(100, 108, 255, 0.5);
    background: linear-gradient(135deg, rgba(100, 108, 255, 0.15), rgba(167, 139, 250, 0.08));
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.7rem;
    color: #a0a0c0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* ============ BUTTON STYLES ============ */
.stButton > button {
    background: linear-gradient(135deg, #6468ff, #a78bfa);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(100, 108, 255, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(100, 108, 255, 0.4);
    background: linear-gradient(135deg, #7c80ff, #b89bff);
}

/* Secondary button */
.stButton > button[data-testid="baseButton-secondary"] {
    background: rgba(255, 255, 255, 0.05);
    box-shadow: none;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: rgba(100, 108, 255, 0.15);
    border-color: rgba(100, 108, 255, 0.4);
}

/* ============ TYPOGRAPHY ============ */
h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #ffffff, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 1rem !important;
    margin-bottom: 0.75rem !important;
}

h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #e2e2f0 !important;
    margin-bottom: 0.5rem !important;
}

p, .stMarkdown {
    color: #c0c0e0;
    line-height: 1.6;
}

/* ============ DATAFRAME STYLING ============ */
[data-testid="stDataFrame"] {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stDataFrame"] table {
    color: #e2e2f0;
}

/* ============ FILE UPLOADER ============ */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.02);
    border: 2px dashed rgba(100, 108, 255, 0.3);
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(100, 108, 255, 0.6);
    background: rgba(100, 108, 255, 0.05);
}

/* ============ EXPANDER ============ */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
}

[data-testid="stExpander"] details {
    background: transparent;
}

/* ============ TABS ============ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.03);
    padding: 0.5rem;
    border-radius: 60px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    color: #a0a0c0;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6468ff, #a78bfa);
    color: white;
}

/* ============ PROGRESS BAR ============ */
.stProgress > div > div {
    background: linear-gradient(90deg, #6468ff, #a78bfa);
}

/* ============ ALERTS ============ */
.stAlert {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #e2e2f0 !important;
}

.stAlert [data-testid="stMarkdown"] {
    color: #e2e2f0 !important;
}

/* ============ BADGES ============ */
.badge-ready {
    background: linear-gradient(135deg, #10b981, #34d399);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    color: white !important;
}

.badge-waiting {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    color: white !important;
}

/* ============ DIVIDER ============ */
hr {
    margin: 1.5rem 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #6468ff, #a78bfa);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #7c80ff, #b89bff);
}

/* ============ CODE BLOCKS ============ */
code, pre {
    background: rgba(0, 0, 0, 0.4);
    border-radius: 8px;
    padding: 0.2rem 0.4rem;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.85rem;
    color: #c4b5fd;
}

/* ============ SELECT BOX ============ */
[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
}

[data-baseweb="select"] * {
    color: #e2e2f0;
}

/* ============ TEXT INPUT ============ */
[data-testid="stTextInput"] input {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    color: #e2e2f0;
}

[data-testid="stTextInput"] input:focus {
    border-color: #6468ff;
    box-shadow: 0 0 0 2px rgba(100, 108, 255, 0.2);
}

/* ============ TEXT AREA ============ */
[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    color: #e2e2f0;
}

/* ============ MULTISELECT ============ */
[data-baseweb="tag"] {
    background: rgba(100, 108, 255, 0.2);
    border-radius: 8px;
}

/* ============ SLIDER ============ */
[data-testid="stSlider"] {
    color: #6468ff;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem !important;
        margin: 0.5rem !important;
    }
    
    h1 {
        font-size: 2rem !important;
    }
    
    .metric-value {
        font-size: 1.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
init_session()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CSV Agent Analyzer")
    st.markdown("### AI-Powered Analysis")
    st.markdown("---")
    
    pages = {
        "🏠 Home": "home",
        "🔍 Data Quality": "quality",
        "🧹 Data Cleaning": "cleaning",
        "📊 Statistical Analysis": "stats",
        "📈 Visualization": "visualization",
        "🤖 Modeling & Evaluation": "modeling",
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
    
    # ── Agent Directory Panel ─────────────────────────────────────────────────
    st.markdown("### 🤖 Agent Directory")
    
    quality_ready = st.session_state.get("quality_report") is not None
    cleaning_ready = st.session_state.get("cleaning_report") is not None
    stats_ready = st.session_state.get("stats_done", False)
    viz_ready = st.session_state.get("viz_done", False)
    modeling_ready = st.session_state.get("modeling_done", False)
    insights_ready = st.session_state.get("insights_text") is not None
    report_ready = st.session_state.get("report_text") is not None
    
    agent_statuses = [
        ("System Orchestrator", "Ready", "⚡"),
        ("Quality Agent", "Ready" if quality_ready else "Waiting", "🔍"),
        ("Cleaning Agent", "Ready" if cleaning_ready else "Waiting", "🧹"),
        ("Statistical Agent", "Ready" if stats_ready else "Waiting", "📊"),
        ("Visuals Agent", "Ready" if viz_ready else "Waiting", "📈"),
        ("Modeling Agent", "Ready" if modeling_ready else "Waiting", "🤖"),
        ("Insights Agent", "Ready" if insights_ready else "Waiting", "💡"),
        ("Report Agent", "Ready" if report_ready else "Waiting", "📄"),
    ]
    
    for name, status, icon in agent_statuses:
        status_color = "color:#34d399; background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.2);" if status == "Ready" else "color:#64748b; background:rgba(148,163,184,0.06); border:1px solid rgba(148,163,184,0.15);"
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.45rem 0.6rem; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 6px; margin-bottom: 0.3rem;">
            <span style="font-size: 0.9rem; margin-right: 0.5rem;">{icon}</span>
            <span style="font-size: 0.78rem; font-weight: 500; color: #94a3b8; flex-grow: 1;">{name}</span>
            <span style="font-size: 0.68rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; {status_color}">{status}</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # ── API Key configurator ──────────────────────────────────────────────────
    from utils.llm import get_api_key
    if not get_api_key():
        st.markdown("### 🔑 OpenRouter API Key")
        api_key_input = st.text_input("Enter Key", type="password", help="Provide your OpenRouter API key to activate the AI Insights and Report Writing agents.")
        if api_key_input:
            st.session_state.openrouter_api_key = api_key_input
            st.rerun()
        st.markdown("---")

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
    elif page == "modeling":
        from pages_content.modeling import render
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
