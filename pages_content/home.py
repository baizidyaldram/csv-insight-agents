import streamlit as st
import pandas as pd
import numpy as np
from utils.session import set_df, is_data_loaded, get_df


def render():
    # Centered Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🤖 CSV Insight Agents</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">
            Multi-agent AI system for automated data analysis and visualization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent Pipeline Section
    st.markdown("### 🔄 Agent Pipeline")
    
    # Create 2 columns for agents
    col1, col2 = st.columns(2, gap="medium")
    
    agents_left = [
        ("🔍", "Data Quality Agent", "Automatic quality assessment with scoring"),
        ("🧹", "Data Cleaning Agent", "Smart cleaning for missing values & duplicates"),
        ("📊", "Statistical Analysis", "Descriptive stats, correlations & distributions"),
    ]
    
    agents_right = [
        ("📈", "Visualization Agent", "Interactive charts powered by Plotly"),
        ("🤖", "Modeling & Evaluation Agent", "ML algorithms, predictions & SHAP analysis"),
        ("💡", "AI Insights Agent", "LLM-powered business insights (OpenRouter)"),
        ("📄", "Report Agent", "Generate comprehensive analysis reports"),
    ]
    
    with col1:
        for icon, title, desc in agents_left:
            st.markdown(f"""
            <div class="agent-card">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div>
                        <h4 style="margin: 0;">{title}</h4>
                        <p style="margin: 0; font-size: 0.8rem;">{desc}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        for icon, title, desc in agents_right:
            st.markdown(f"""
            <div class="agent-card">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div>
                        <h4 style="margin: 0;">{title}</h4>
                        <p style="margin: 0; font-size: 0.8rem;">{desc}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload Section - Centered
    st.markdown("### 📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload any CSV file - sales data, surveys, financial records, etc.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            set_df(df, uploaded_file.name)
            st.balloons()
            st.success(f"✅ Successfully loaded **{uploaded_file.name}**!")
            # REMOVED: st.rerun() - this was causing the freeze
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Centered "or"
    st.markdown("<div style='text-align: center; margin: 1rem 0'>— or —</div>", unsafe_allow_html=True)
    
    # Centered button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📦 Load Sample Dataset", use_container_width=True):
            try:
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
                st.balloons()
                st.success("✅ Sample dataset loaded (50 rows)!")
                # REMOVED: st.rerun() - this was causing the freeze
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Data Preview Section (when data is loaded)
    if is_data_loaded():
        df = get_df()
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        
        # Metrics in centered columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Rows</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Columns</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            numeric_count = df.select_dtypes(include="number").shape[1]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{numeric_count}</div>
                <div class="metric-label">Numeric Cols</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{missing_pct:.1f}%</div>
                <div class="metric-label">Missing Data</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Preview
        st.markdown("#### 👀 Data Preview (First 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Quick Actions
        st.markdown("---")
        st.markdown("#### 🚀 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🔍 Quality Check", key="btn_quality_home", use_container_width=True):
                st.session_state.current_page = "quality"
                st.rerun()
        with col2:
            if st.button("🧹 Clean Data", key="btn_cleaning_home", use_container_width=True):
                st.session_state.current_page = "cleaning"
                st.rerun()
        with col3:
            if st.button("📊 Statistics", key="btn_stats_home", use_container_width=True):
                st.session_state.current_page = "stats"
                st.rerun()
        with col4:
            if st.button("🤖 Modeling", key="btn_modeling_home", use_container_width=True):
                st.session_state.current_page = "modeling"
                st.rerun()
        
        st.info("💡 **Pro Tip:** Navigate through the agents in order for the best analysis experience!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 0 0; color: rgba(255,255,255,0.4); font-size: 0.75rem;">
        Powered by Streamlit | Plotly Visualizations | XGBoost | OpenRouter AI
    </div>
    """, unsafe_allow_html=True)
