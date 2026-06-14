import streamlit as st
import pandas as pd
import numpy as np
from utils.session import set_df, is_data_loaded, get_df


def render():
    # Custom CSS
    st.markdown("""
    <style>
    .hero-section {
        text-align: center;
        padding: 1rem 0 1rem 0;
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
    .success-banner {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">CSV Insight Agents</div>
        <div class="hero-subtitle">AI-powered data analysis with multi-agent orchestration</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # SECTION 1: UPLOAD DATA
    # ============================================================
    st.markdown("### 📁 Upload Your Data")

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
                st.markdown(f"""
                <div class="success-banner">
                    ✅ <strong>Successfully loaded {uploaded_file.name}!</strong><br>
                    <span style="font-size: 0.8rem;">{df.shape[0]:,} rows × {df.shape[1]} columns</span>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("📦 Load Sample Dataset", key="load_sample_btn", use_container_width=True):
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
                st.success("✅ Sample dataset loaded! (50 rows)")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # ============================================================
    # SECTION 2: AGENT PIPELINE ARCHITECTURE
    # ============================================================
    if is_data_loaded():
        df = get_df()
        
        st.markdown("## Agent Pipeline Architecture")
        st.markdown("_Follow these steps in order for complete analysis_")

        # Row 1
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">🔍</div>
                <div style="font-weight: 700;">1. Data Quality</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Quality scoring & validation</div>
            </div>
            """, unsafe_allow_html=True)
            # Use query params or direct link - simpler approach
            if st.button("Go to Quality →", key="goto_quality", use_container_width=True):
                st.session_state.current_page = "quality"
                st.rerun()
        
        with col2:
            st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">🧹</div>
                <div style="font-weight: 700;">2. Data Cleaning</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Missing values & duplicates</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Cleaning →", key="goto_cleaning", use_container_width=True):
                st.session_state.current_page = "cleaning"
                st.rerun()
        
        with col3:
            st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">📊</div>
                <div style="font-weight: 700;">3. Statistics</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Descriptive & correlations</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Statistics →", key="goto_stats", use_container_width=True):
                st.session_state.current_page = "stats"
                st.rerun()
        
        with col4:
            st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">📈</div>
                <div style="font-weight: 700;">4. Visualization</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Interactive charts</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Visualization →", key="goto_viz", use_container_width=True):
                st.session_state.current_page = "visualization"
                st.rerun()

        # Row 2
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">🤖</div>
                <div style="font-weight: 700;">5. Modeling</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">ML algorithms & predictions</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Modeling →", key="goto_modeling", use_container_width=True):
                st.session_state.current_page = "modeling"
                st.rerun()
        
        with col2:
            st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">💡</div>
                <div style="font-weight: 700;">6. AI Insights</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">LLM-powered analysis</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Insights →", key="goto_insights", use_container_width=True):
                st.session_state.current_page = "insights"
                st.rerun()
        
        with col3:
            st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">📄</div>
                <div style="font-weight: 700;">7. Report</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Comprehensive export</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Report →", key="goto_report", use_container_width=True):
                st.session_state.current_page = "report"
                st.rerun()
        
        with col4:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">✅</div>
                <div style="font-weight: 700;">Data Loaded</div>
                <div style="font-size: 0.7rem; color: #10B981;">{df.shape[0]:,} rows, {df.shape[1]} cols</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data Preview
        with st.expander("📊 Data Preview (First 5 rows)", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)
    
    else:
        st.info("📌 **Ready to start?** Upload a CSV file above to begin the analysis pipeline.")
        
        # Show pipeline preview
        st.markdown("""
        ### 🧠 Analysis Pipeline (Available after upload)
        
        | Step | Tool | Description |
        |------|------|-------------|
        | 1 | 🔍 Data Quality | Automatic quality scoring |
        | 2 | 🧹 Data Cleaning | Handle missing values & duplicates |
        | 3 | 📊 Statistics | Descriptive stats & correlations |
        | 4 | 📈 Visualization | Interactive Plotly charts |
        | 5 | 🤖 Modeling | XGBoost, LightGBM, SHAP analysis |
        | 6 | 💡 AI Insights | LLM-powered business insights |
        | 7 | 📄 Report | Comprehensive downloadable report |
        """, unsafe_allow_html=False)

    # ============================================================
    # SECTION 3: MACHINE LEARNING MODELING STRUCTURE
    # ============================================================
    st.markdown("## Machine Learning Modeling Structure")

    with st.expander("📊 View Complete Modeling Pipeline", expanded=False):
        st.markdown("""
        ### 🎯 Modeling Pipeline Steps
        
        | Step | Description |
        |------|-------------|
        | 1 | **Target Selection** - Choose the column to predict |
        | 2 | **Feature Selection** - Select predictor variables |
        | 3 | **Data Preprocessing** - Handle missing values, encode categories, scale features |
        | 4 | **Algorithm Selection** - XGBoost, LightGBM, Random Forest, etc. |
        | 5 | **Hyperparameter Tuning** - Automatic optimization with Optuna |
        | 6 | **Cross-Validation** - K-Fold validation to prevent overfitting |
        | 7 | **Performance Evaluation** - Accuracy, F1, ROC-AUC / R², MAE, RMSE |
        | 8 | **Model Explainability** - SHAP values, Feature Importance |
        | 9 | **Predictions Playground** - Test what-if scenarios |
        """, unsafe_allow_html=False)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.2rem;">🏷️</span>
                <span style="font-weight: 700;">Classification Models</span>
            </div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                • XGBoost Classifier<br>
                • LightGBM Classifier<br>
                • Random Forest Classifier<br>
                • Logistic Regression<br>
                • Gradient Boosting Classifier
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.7rem; color: #10B981;">
                ✓ SMOTE for imbalanced data
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.2rem;">📈</span>
                <span style="font-weight: 700;">Regression Models</span>
            </div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                • XGBoost Regressor<br>
                • LightGBM Regressor<br>
                • Random Forest Regressor<br>
                • Linear Regression<br>
                • Ridge Regression
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.7rem; color: #10B981;">
                ✓ Automatic task detection
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 0.7rem;">
        Powered by Streamlit • XGBoost • LightGBM • SHAP • OpenRouter AI
    </div>
    """, unsafe_allow_html=True)
