import streamlit as st
import pandas as pd
import numpy as np
from utils.session import set_df, is_data_loaded, get_df


def render():
    # Hero Section - Clean and Minimal
    st.markdown("""
    <style>
    .hero-section {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
    }
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .pipeline-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .pipeline-card:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
    }
    .pipeline-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .pipeline-name {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-primary);
    }
    .pipeline-desc {
        font-size: 0.7rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    .upload-area {
        background: var(--bg-card);
        border: 2px dashed var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10B981;
    }
    .stat-label {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    .model-structure {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    .model-step {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-color);
    }
    .model-step:last-child {
        border-bottom: none;
    }
    .step-number {
        width: 28px;
        height: 28px;
        background: #3B82F6;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
    }
    .step-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--text-primary);
    }
    .step-desc {
        font-size: 0.75rem;
        color: var(--text-secondary);
    }
    .action-buttons {
        display: flex;
        gap: 0.75rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    @media (max-width: 768px) {
        .pipeline-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
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

    # ============================================================
    # 8-AGENT PIPELINE STRUCTURE (Clearly Organized)
    # ============================================================
    st.markdown("### 🔄 Agent Pipeline Architecture")
    st.markdown("_Follow these steps in order for complete analysis_")

    # Row 1: Data Preparation Agents
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="pipeline-card">
            <div class="pipeline-icon">🔍</div>
            <div class="pipeline-name">1. Data Quality</div>
            <div class="pipeline-desc">Quality scoring & validation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-card">
            <div class="pipeline-icon">🧹</div>
            <div class="pipeline-name">2. Data Cleaning</div>
            <div class="pipeline-desc">Missing values & duplicates</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pipeline-card">
            <div class="pipeline-icon">📊</div>
            <div class="pipeline-name">3. Statistics</div>
            <div class="pipeline-desc">Descriptive & correlations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="pipeline-card">
            <div class="pipeline-icon">📈</div>
            <div class="pipeline-name">4. Visualization</div>
            <div class="pipeline-desc">Interactive charts</div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Advanced Analytics Agents
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="pipeline-card">
            <div class="pipeline-icon">🤖</div>
            <div class="pipeline-name">5. Modeling</div>
            <div class="pipeline-desc">ML algorithms & predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-card">
            <div class="pipeline-icon">💡</div>
            <div class="pipeline-name">6. AI Insights</div>
            <div class="pipeline-desc">LLM-powered analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pipeline-card">
            <div class="pipeline-icon">📄</div>
            <div class="pipeline-name">7. Report</div>
            <div class="pipeline-desc">Comprehensive export</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Status indicator
        if is_data_loaded():
            st.markdown("""
            <div class="pipeline-card" style="border-color: #10B981;">
                <div class="pipeline-icon">✅</div>
                <div class="pipeline-name">Data Loaded</div>
                <div class="pipeline-desc">Ready for analysis</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pipeline-card">
                <div class="pipeline-icon">📁</div>
                <div class="pipeline-name">Upload Data</div>
                <div class="pipeline-desc">Start here</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # MODELING STRUCTURE SECTION (NEW - Clearly Explained)
    # ============================================================
    st.markdown("### 🧠 Machine Learning Modeling Structure")

    with st.expander("📊 View Complete Modeling Pipeline", expanded=False):
        st.markdown("""
        <div class="model-structure">
            <h4 style="margin-bottom: 1rem; color: #10B981;">🎯 Target Selection</h4>
            <div class="model-step">
                <div class="step-number">1</div>
                <div>
                    <div class="step-title">Choose Target Variable</div>
                    <div class="step-desc">Select the column you want to predict (Y)</div>
                </div>
            </div>
            
            <h4 style="margin: 1rem 0 0.5rem 0; color: #10B981;">📋 Feature Engineering</h4>
            <div class="model-step">
                <div class="step-number">2</div>
                <div>
                    <div class="step-title">Feature Selection</div>
                    <div class="step-desc">Choose predictor variables (X) - automatic or manual</div>
                </div>
            </div>
            <div class="model-step">
                <div class="step-number">3</div>
                <div>
                    <div class="step-title">Data Preprocessing</div>
                    <div class="step-desc">Handling missing values, encoding categories, scaling</div>
                </div>
            </div>
            
            <h4 style="margin: 1rem 0 0.5rem 0; color: #10B981;">🤖 Model Training</h4>
            <div class="model-step">
                <div class="step-number">4</div>
                <div>
                    <div class="step-title">Algorithm Selection</div>
                    <div class="step-desc">XGBoost, LightGBM, Random Forest, Linear Models, etc.</div>
                </div>
            </div>
            <div class="model-step">
                <div class="step-number">5</div>
                <div>
                    <div class="step-title">Hyperparameter Tuning</div>
                    <div class="step-desc">Automatic optimization using Optuna/RandomizedSearch</div>
                </div>
            </div>
            <div class="model-step">
                <div class="step-number">6</div>
                <div>
                    <div class="step-title">Cross-Validation</div>
                    <div class="step-desc">K-Fold validation to prevent overfitting</div>
                </div>
            </div>
            
            <h4 style="margin: 1rem 0 0.5rem 0; color: #10B981;">📈 Evaluation</h4>
            <div class="model-step">
                <div class="step-number">7</div>
                <div>
                    <div class="step-title">Performance Metrics</div>
                    <div class="step-desc">Accuracy, F1, ROC-AUC (classification) / R², MAE, RMSE (regression)</div>
                </div>
            </div>
            <div class="model-step">
                <div class="step-number">8</div>
                <div>
                    <div class="step-title">Model Explainability</div>
                    <div class="step-desc">SHAP values, Feature Importance analysis</div>
                </div>
            </div>
            
            <h4 style="margin: 1rem 0 0.5rem 0; color: #10B981;">🎮 Predictions</h4>
            <div class="model-step">
                <div class="step-number">9</div>
                <div>
                    <div class="step-title">Interactive Playground</div>
                    <div class="step-desc">Test what-if scenarios with real-time predictions</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Model Type Reference
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-structure">
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
        <div class="model-structure">
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

    st.markdown("---")

    # ============================================================
    # UPLOAD SECTION
    # ============================================================
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
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Sample Data Button
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
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ============================================================
    # DATA PREVIEW (When data is loaded)
    # ============================================================
    if is_data_loaded():
        df = get_df()
        st.markdown("---")
        st.markdown("### 📊 Current Dataset Overview")
        
        # Stats Grid
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
                <div class="stat-label">Numeric Cols</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{missing_pct:.1f}%</div>
                <div class="stat-label">Missing Data</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Preview
        st.markdown("#### 👀 Data Preview (First 5 rows)")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Quick Action Buttons
        st.markdown("#### 🚀 Quick Navigation")
        
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
            if st.button("🤖 Modeling", use_container_width=True):
                st.session_state.current_page = "modeling"
                st.rerun()
        
        # Pipeline Progress Indicator
        st.markdown("---")
        st.markdown("#### 📋 Analysis Pipeline Status")
        
        steps_completed = []
        steps = [
            ("quality", "Data Quality", st.session_state.get("quality_report") is not None),
            ("cleaning", "Data Cleaning", st.session_state.get("cleaning_report") is not None),
            ("stats", "Statistical Analysis", st.session_state.get("stats_done", False)),
            ("visualization", "Visualization", st.session_state.get("viz_done", False)),
            ("modeling", "ML Modeling", st.session_state.get("modeling_done", False)),
            ("insights", "AI Insights", st.session_state.get("insights_text") is not None),
            ("report", "Report", st.session_state.get("report_text") is not None),
        ]
        
        completed_count = sum(1 for _, _, done in steps if done)
        progress = completed_count / len(steps) if steps else 0
        
        st.progress(progress)
        st.caption(f"📌 {completed_count}/{len(steps)} steps completed")
        
        # Show completed steps
        cols = st.columns(7)
        for col, (_, name, done) in zip(cols, steps):
            icon = "✅" if done else "⏳"
            color = "#10B981" if done else "#64748B"
            col.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 1.2rem;">{icon}</div>
                <div style="font-size: 0.65rem; color: {color};">{name}</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 0 0; color: var(--text-secondary); font-size: 0.7rem;">
        Powered by Streamlit | XGBoost • LightGBM • SHAP • OpenRouter AI
    </div>
    """, unsafe_allow_html=True)
