import streamlit as st
from utils.session import init_session, get_df, is_data_loaded
import pandas as pd
import sys
import os
import psutil
from datetime import datetime

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Insight Agents | AI-Powered Data Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Modern Theme CSS (Slate & Emerald) ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* CSS Variables for Theming */
:root {
    --bg-primary: #0F172A;
    --bg-secondary: #1E293B;
    --bg-card: rgba(30, 41, 59, 0.7);
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --accent-success: #10B981;
    --accent-success-hover: #059669;
    --accent-warning: #F59E0B;
    --accent-error: #EF4444;
    --accent-info: #3B82F6;
    --border-color: #334155;
    --border-radius-sm: 8px;
    --border-radius-md: 12px;
    --border-radius-lg: 16px;
    --transition-speed: 200ms;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* Light Theme */
[data-theme="light"] {
    --bg-primary: #F8FAFC;
    --bg-secondary: #FFFFFF;
    --bg-card: rgba(255, 255, 255, 0.9);
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --border-color: #E2E8F0;
}

/* Global Styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.stApp {
    background: var(--bg-primary);
    transition: background var(--transition-speed) ease;
}

/* Main content container */
.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Glass morphism cards */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    padding: 1.25rem;
    transition: all var(--transition-speed) ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

/* Hide Streamlit Branding */
#MainMenu, footer, header { visibility: hidden; }

/* ============ SIDEBAR STYLING ============ */
[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    transition: transform var(--transition-speed) ease;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Sidebar navigation buttons */
[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-md);
    text-align: left;
    padding: 0.65rem 1rem;
    transition: all var(--transition-speed) ease;
    font-weight: 500;
    width: 100%;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(59, 130, 246, 0.1);
    border-color: var(--accent-info);
    transform: translateX(4px);
}

[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--accent-info), var(--accent-success));
    border: none;
    color: white !important;
}

/* Metric Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-md);
    padding: 1rem;
    text-align: center;
    transition: all var(--transition-speed) ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-info);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-success);
}

.metric-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* Primary Button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-info), var(--accent-success));
    color: white;
    border: none;
    border-radius: var(--border-radius-md);
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all var(--transition-speed) ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

/* Status Badges */
.badge-success {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-success);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-warning);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}

/* Progress Timeline */
.progress-timeline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1rem 0;
}

.timeline-step {
    text-align: center;
    flex: 1;
    position: relative;
}

.timeline-step .step-icon {
    width: 40px;
    height: 40px;
    background: var(--bg-secondary);
    border: 2px solid var(--border-color);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 0.5rem;
    transition: all var(--transition-speed) ease;
}

.timeline-step.completed .step-icon {
    background: var(--accent-success);
    border-color: var(--accent-success);
    color: white;
}

.timeline-step.active .step-icon {
    border-color: var(--accent-info);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

/* Data Table Styling */
[data-testid="stDataFrame"] {
    background: var(--bg-card);
    border-radius: var(--border-radius-md);
    border: 1px solid var(--border-color);
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 2px dashed var(--border-color);
    border-radius: var(--border-radius-lg);
    padding: 2rem;
    transition: all var(--transition-speed) ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-info);
    background: rgba(59, 130, 246, 0.05);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--accent-info);
    border-radius: 10px;
}

/* Toast Notification */
.toast-notification {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: var(--accent-success);
    color: white;
    padding: 0.75rem 1.25rem;
    border-radius: var(--border-radius-md);
    animation: slideIn 0.3s ease;
    z-index: 1000;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Responsive Design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem !important;
    }
    
    .metric-value {
        font-size: 1.5rem;
    }
    
    [data-testid="stSidebar"] {
        transform: translateX(-100%);
        position: fixed;
        z-index: 100;
    }
    
    [data-testid="stSidebar"].open {
        transform: translateX(0);
    }
}
</style>

<script>
// Theme initialization
function initTheme() {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('theme');
    const theme = savedTheme || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('[data-testid="stTextInput"] input');
        if (searchInput) searchInput.focus();
    }
});

initTheme();
</script>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
init_session()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CSV Agent Analyzer")
    st.markdown("### AI-Powered Analysis")
    st.markdown("---")
    
    # Theme toggle in sidebar
    col_theme1, col_theme2 = st.columns(2)
    with col_theme1:
        if st.button("🌙 Dark", use_container_width=True, key="theme_dark"):
            st.markdown('<script>document.documentElement.setAttribute("data-theme", "dark"); localStorage.setItem("theme", "dark");</script>', unsafe_allow_html=True)
            st.rerun()
    with col_theme2:
        if st.button("☀️ Light", use_container_width=True, key="theme_light"):
            st.markdown('<script>document.documentElement.setAttribute("data-theme", "light"); localStorage.setItem("theme", "light");</script>', unsafe_allow_html=True)
            st.rerun()
    
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
        ("System Orchestrator", "Ready" if is_data_loaded() else "Waiting", "⚡"),
        ("Quality Agent", "Ready" if quality_ready else "Waiting", "🔍"),
        ("Cleaning Agent", "Ready" if cleaning_ready else "Waiting", "🧹"),
        ("Statistical Agent", "Ready" if stats_ready else "Waiting", "📊"),
        ("Visuals Agent", "Ready" if viz_ready else "Waiting", "📈"),
        ("Modeling Agent", "Ready" if modeling_ready else "Waiting", "🤖"),
        ("Insights Agent", "Ready" if insights_ready else "Waiting", "💡"),
        ("Report Agent", "Ready" if report_ready else "Waiting", "📄"),
    ]
    
    for name, status, icon in agent_statuses:
        badge_class = "badge-success" if status == "Ready" else "badge-warning"
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem; margin-bottom: 0.25rem;">
            <span style="font-size: 0.9rem;">{icon}</span>
            <span style="font-size: 0.8rem; flex: 1; margin-left: 0.5rem;">{name}</span>
            <span class="{badge_class}" style="font-size: 0.65rem;">{status}</span>
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
        st.markdown(f'<span class="badge-success">✓ Data Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"### {df.shape[0]:,}")
        st.caption("rows")
        st.markdown(f"### {df.shape[1]}")
        st.caption("columns")
        
        with st.expander("📊 Quick Preview"):
            st.dataframe(df.head(3), use_container_width=True)
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.caption(f"💾 Memory: {memory_mb:.1f} MB")
    else:
        st.markdown(f'<span class="badge-warning">⏳ No Data</span>', unsafe_allow_html=True)
        st.caption("Upload a CSV to begin")
    
    st.markdown("---")
    st.caption("💡 Press **Ctrl+K** to search")

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
