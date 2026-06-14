import streamlit as st
import pandas as pd
import numpy as np
from utils.session import set_df, is_data_loaded, get_df


def navigate_to(page):
    """Helper function to navigate to a different page"""
    st.session_state.current_page = page
    st.rerun()


def render():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .hero-section {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    .step-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
        cursor: pointer;
        margin-bottom: 0.75rem;
    }
    .step-card:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
        background: rgba(59, 130, 246, 0.05);
    }
    .step-number {
        display: inline-block;
        width: 28px;
        height: 28px;
        background: #3B82F6;
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        line-height: 28px;
        margin-bottom: 0.5rem;
    }
    .step-icon {
        font-size: 1.8rem;
        margin-bottom: 0.25rem;
    }
    .step-name {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-primary);
    }
    .step-desc {
        font-size: 0.7rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    .step-status {
        font-size: 0.65rem;
        margin-top: 0.5rem;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        display: inline-block;
    }
    .status-done {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
    }
    .status-pending {
        background: rgba(100, 116, 139, 0.15);
        color: #94A3B8;
    }
    .upload-area {
        background: var(--bg-card);
        border: 2px dashed var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0 1rem 0;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10B981;
    }
    .stat-label {
        font-size: 0.65rem;
        color: var(--text-secondary);
        text-transform: uppercase;
    }
    .progress-section {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .quick-actions {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">CSV Insight Agents</div>
        <div class="hero-subtitle">Multi-agent AI system for automated data analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # STEP 1: UPLOAD DATA (Always first, always visible)
    # ============================================================
    st.markdown("### 📁 Step 1: Upload Your Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
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
                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("📦 Sample Dataset", use_container_width=True):
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
                st.success("✅ Sample dataset loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # ============================================================
    # STEPS 2-8: ANALYSIS PIPELINE (Only show if data is loaded)
    # ============================================================
    if is_data_loaded():
        df = get_df()
        
        # Dataset Quick Stats
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{df.shape[0]:,}</div>
                <div class="stat-label">Rows</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{df.shape[1]}</div>
                <div class="stat-label">Columns</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            numeric_count = df.select_dtypes(include="number").shape[1]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{numeric_count}</div>
                <div class="stat-label">Numeric</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{missing_pct:.1f}%</div>
                <div class="stat-label">Missing</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔬 Step 2-8: Analysis Pipeline")
        st.markdown("_Click any card to navigate to that analysis tool_")
        
        # Row 1: Quality, Cleaning, Statistics
        col1, col2, col3 = st.columns(3)
        
        # Get completion statuses
        quality_done = st.session_state.get("quality_report") is not None
        cleaning_done = st.session_state.get("cleaning_report") is not None
        stats_done = st.session_state.get("stats_done", False)
        viz_done = st.session_state.get("viz_done", False)
        modeling_done = st.session_state.get("modeling_done", False)
        insights_done = st.session_state.get("insights_text") is not None
        report_done = st.session_state.get("report_text") is not None
        
        with col1:
            status_class = "status-done" if quality_done else "status-pending"
            status_text = "✓ Completed" if quality_done else "Pending"
            st.markdown(f"""
            <div class="step-card" onclick="window.location.href='?page=quality'">
                <div class="step-number">2</div>
                <div class="step-icon">🔍</div>
                <div class="step-name">Data Quality</div>
                <div class="step-desc">Quality scoring & validation</div>
                <div class="step-status {status_class}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Quality", key="btn_quality", use_container_width=True):
                navigate_to("quality")
        
        with col2:
            status_class = "status-done" if cleaning_done else "status-pending"
            status_text = "✓ Completed" if cleaning_done else "Pending"
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">3</div>
                <div class="step-icon">🧹</div>
                <div class="step-name">Data Cleaning</div>
                <div class="step-desc">Missing values & duplicates</div>
                <div class="step-status {status_class}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Cleaning", key="btn_cleaning", use_container_width=True):
                navigate_to("cleaning")
        
        with col3:
            status_class = "status-done" if stats_done else "status-pending"
            status_text = "✓ Completed" if stats_done else "Pending"
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">4</div>
                <div class="step-icon">📊</div>
                <div class="step-name">Statistical Analysis</div>
                <div class="step-desc">Descriptive & correlations</div>
                <div class="step-status {status_class}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Statistics", key="btn_stats", use_container_width=True):
                navigate_to("stats")
        
        # Row 2: Visualization, Modeling, AI Insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_class = "status-done" if viz_done else "status-pending"
            status_text = "✓ Completed" if viz_done else "Pending"
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">5</div>
                <div class="step-icon">📈</div>
                <div class="step-name">Visualization</div>
                <div class="step-desc">Interactive charts</div>
                <div class="step-status {status_class}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Visualization", key="btn_viz", use_container_width=True):
                navigate_to("visualization")
        
        with col2:
            status_class = "status-done" if modeling_done else "status-pending"
            status_text = "✓ Completed" if modeling_done else "Pending"
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">6</div>
                <div class="step-icon">🤖</div>
                <div class="step-name">Modeling & Evaluation</div>
                <div class="step-desc">XGBoost, LightGBM, SHAP</div>
                <div class="step-status {status_class}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Modeling", key="btn_modeling", use_container_width=True):
                navigate_to("modeling")
        
        with col3:
            status_class = "status-done" if insights_done else "status-pending"
            status_text = "✓ Completed" if insights_done else "Pending"
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">7</div>
                <div class="step-icon">💡</div>
                <div class="step-name">AI Insights</div>
                <div class="step-desc">LLM-powered analysis</div>
                <div class="step-status {status_class}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Insights", key="btn_insights", use_container_width=True):
                navigate_to("insights")
        
        # Row 3: Report (Full width)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            status_class = "status-done" if report_done else "status-pending"
            status_text = "✓ Completed" if report_done else "Pending"
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">8</div>
                <div class="step-icon">📄</div>
                <div class="step-name">Generate Report</div>
                <div class="step-desc">Comprehensive analysis export</div>
                <div class="step-status {status_class}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Report", key="btn_report", use_container_width=True):
                navigate_to("report")
        
        # Progress Summary
        st.markdown("---")
        st.markdown("### 📈 Pipeline Progress")
        
        completed = sum([quality_done, cleaning_done, stats_done, viz_done, modeling_done, insights_done, report_done])
        total = 7
        progress = completed / total
        
        st.progress(progress)
        st.caption(f"✅ {completed}/{total} analysis steps completed")
        
        # Quick Data Preview
        with st.expander("📋 Data Preview (First 5 rows)", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)
        
    else:
        # No data loaded - show prompt
        st.info("📌 **Ready to start?** Upload a CSV file above to begin the analysis pipeline.")
        
        # Show sample of what the pipeline looks like
        st.markdown("""
        ### 🧠 What happens after you upload?
        
        | Step | Tool | Description |
        |------|------|-------------|
        | 1 | 📁 Upload | Load your CSV data |
        | 2 | 🔍 Data Quality | Automatic quality scoring |
        | 3 | 🧹 Data Cleaning | Handle missing values & duplicates |
        | 4 | 📊 Statistics | Descriptive stats & correlations |
        | 5 | 📈 Visualization | Interactive Plotly charts |
        | 6 | 🤖 Modeling | XGBoost, LightGBM, SHAP analysis |
        | 7 | 💡 AI Insights | LLM-powered business insights |
        | 8 | 📄 Report | Comprehensive downloadable report |
        """, unsafe_allow_html=False)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 0.7rem;">
        Powered by Streamlit • XGBoost • LightGBM • SHAP • OpenRouter AI
    </div>
    """, unsafe_allow_html=True)
