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

# ── Modern CSS with Beautiful Background ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Poppins', 'Outfit', sans-serif;
}

/* Beautiful Animated Background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
    position: relative;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(119, 198, 255, 0.2) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* Main content area - Perfectly Centered */
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

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar - Glass morphism */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar content styling */
[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff;
}

/* Custom sidebar toggle */
[data-testid="collapsedControl"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 0 12px 12px 0;
    padding: 8px 12px;
}

/* Modern Cards with Gradient */
.glass-card, .agent-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 24px;
    padding: 1.5rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.glass-card:hover, .agent-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    border-color: rgba(255, 255, 255, 0.4);
}

.agent-card h4 {
    background: linear-gradient(135deg, #fff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}

.agent-card p {
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.85rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.05));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.1));
}

.metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card .label {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.7);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}

/* Upload Area */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 2px dashed rgba(255, 255, 255, 0.3);
    border-radius: 24px;
    padding: 2rem;
}

[data-testid="stFileUploader"]:hover {
    border-color: #fff;
    background: rgba(255, 255, 255, 0.15);
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
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    background: linear-gradient(135deg, #764ba2, #667eea);
}

/* Status Badges */
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

/* Headers */
h1, h2, h3 {
    background: linear-gradient(135deg, #ffffff, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 3rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
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
    font-weight: 500;
    color: rgba(255, 255, 255, 0.7);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
    margin: 2rem 0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

/* Info/Warning/Success boxes */
.stAlert {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    color: white !important;
}

/* Select boxes */
[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
}

/* Text area */
[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 16px;
    color: white;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

[data-testid="stMetricLabel"] {
    color: rgba(255, 255, 255, 0.8) !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
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
    
    # Data status
    if is_data_loaded():
        df = get_df()
        st.markdown(f'<span class="badge-ready">✓ Data Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"### {df.shape[0]:,}")
        st.caption("rows")
        st.markdown(f"### {df.shape[1]}")
        st.caption("columns")
        
        with st.expander("📊 Quick Preview"):
            st.dataframe(df.head(5), use_container_width=True)
    else:
        st.markdown('<span class="badge-waiting">⏳ No data yet</span>', unsafe_allow_html=True)
        st.caption("Upload a CSV on the Home page")

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
