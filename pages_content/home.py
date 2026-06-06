import streamlit as st
import pandas as pd
import numpy as np
from utils.session import set_df, is_data_loaded, get_df


def render():
    # Animated Hero Section
    st.markdown("""
    <style>
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-section {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(100,108,255,0.15), rgba(167,139,250,0.15));
        border-radius: 32px;
        margin-bottom: 2rem;
        animation: slideIn 0.6s ease-out;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #a78bfa, #6468ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: float 3s ease-in-out infinite;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.7);
        margin-bottom: 1rem;
    }
    
    .stats-badge {
        display: inline-flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .stat-pill {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.2);
    }
    </style>
    
    <div class="hero-section">
        <div class="hero-title">
            🤖 CSV Insight Agents
        </div>
        <div class="hero-subtitle">
            Multi-agent AI system for automated data analysis and visualization
        </div>
        <div class="stats-badge">
            <span class="stat-pill">⚡ 6 Intelligent Agents</span>
            <span class="stat-pill">🎨 Interactive Visualizations</span>
            <span class="stat-pill">🤖 AI-Powered Insights</span>
            <span class="stat-pill">📊 Real-time Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent Pipeline Section with Enhanced Cards
    st.markdown("### 🚀 Agent Pipeline")
    
    # Create 2 columns for agents with better spacing
    col1, col2 = st.columns(2, gap="large")
    
    # Enhanced agent cards with gradients and icons
    agents_left = [
        ("🔍", "Data Quality Agent", "Automatic quality assessment with scoring", "#10b981"),
        ("🧹", "Data Cleaning Agent", "Smart cleaning for missing values & duplicates", "#f59e0b"),
        ("📊", "Statistical Analysis", "Descriptive stats, correlations & distributions", "#3b82f6"),
    ]
    
    agents_right = [
        ("📈", "Visualization Agent", "Interactive charts powered by Plotly", "#8b5cf6"),
        ("💡", "AI Insights Agent", "LLM-powered business insights", "#ec4899"),
        ("📄", "Report Agent", "Generate comprehensive analysis reports", "#06b6d4"),
    ]
    
    with col1:
        for icon, title, desc, color in agents_left:
            st.markdown(f"""
            <div class="agent-card" style="border-left: 4px solid {color};">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 2rem; animation: pulse 2s ease-in-out infinite;">{icon}</div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0; font-size: 1rem;">{title}</h4>
                        <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: rgba(255,255,255,0.6);">{desc}</p>
                    </div>
                    <div style="font-size: 1.2rem;">→</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        for icon, title, desc, color in agents_right:
            st.markdown(f"""
            <div class="agent-card" style="border-left: 4px solid {color};">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 2rem; animation: pulse 2s ease-in-out infinite;">{icon}</div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0; font-size: 1rem;">{title}</h4>
                        <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: rgba(255,255,255,0.6);">{desc}</p>
                    </div>
                    <div style="font-size: 1.2rem;">→</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload Section - Enhanced with Drag & Drop Feel
    st.markdown("### 📁 Upload Your Data")
    
    # Create a styled upload area
    upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])
    
    with upload_col2:
        uploaded_file = st.file_uploader(
            "",
            type=["csv"],
            help="Upload any CSV file - sales data, surveys, financial records, etc.",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                set_df(df, uploaded_file.name)
                st.success(f"✅ Successfully loaded **{uploaded_file.name}**!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Divider with "or"
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0; position: relative;">
            <div style="position: relative; display: inline-block; padding: 0 1rem; background: rgba(18,18,35,0.7);">
                — or —
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📦 Load Sample Dataset", use_container_width=True):
            try:
                # Create sample dataset
                np.random.seed(42)
                sample_df = pd.DataFrame({
                    'order_id': range(1001, 1051),
                    'order_date': pd.date_range('2024-01-01', periods=50, freq='D'),
                    'customer_name': [f'Customer_{i}' for i in range(1, 51)],
                    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], 50),
                    'quantity': np.random.randint(1, 10, 50),
                    'price': np.random.uniform(10, 500, 50).round(2),
                    'region': np.random.choice(['North', 'South', 'East', 'West'], 50),
                })
                sample_df['total_amount'] = (sample_df['quantity'] * sample_df['price']).round(2)
                set_df(sample_df, "sample_orders.csv")
                st.success("✅ Sample dataset loaded (50 rows)!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Data Preview Section (when data is loaded)
    if is_data_loaded():
        df = get_df()
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        
        # Enhanced metrics with better styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">📊</div>
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">📋</div>
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_count = df.select_dtypes(include="number").shape[1]
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">🔢</div>
                <div class="metric-value">{numeric_count}</div>
                <div class="metric-label">Numeric Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
            missing_color = "#f59e0b" if missing_pct > 0 else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">⚠️</div>
                <div class="metric-value" style="color: {missing_color};">{missing_pct:.1f}%</div>
                <div class="metric-label">Missing Data</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Preview with better styling
        st.markdown("#### 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Quick Actions with better buttons
        st.markdown("---")
        st.markdown("#### 🚀 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Quality Check", use_container_width=True):
                st.session_state.current_page = "quality"
                st.rerun()
        
        with col2:
            if st.button("🧹 Clean Data", use_container_width=True):
                st.session_state.current_page = "cleaning"
                st.rerun()
        
        with col3:
            if st.button("📊 Statistics", use_container_width=True):
                st.session_state.current_page = "stats"
                st.rerun()
        
        with col4:
            if st.button("📈 Visualize", use_container_width=True):
                st.session_state.current_page = "visualization"
                st.rerun()
        
        # Pro Tip with better styling
        st.info("💡 **Pro Tip:** Navigate through the 6 agents in order for the best analysis experience!")
    
    else:
        # Empty state with better design
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📂</div>
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; background: linear-gradient(135deg, #fff, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                No Data Loaded
            </div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6);">
                Upload a CSV file or load the sample dataset to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with additional info
    st.markdown("---")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem;">⚡</div>
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">Powered by Streamlit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem;">🎨</div>
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">Plotly Visualizations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem;">🤖</div>
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">OpenRouter AI</div>
        </div>
        """, unsafe_allow_html=True)
