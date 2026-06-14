import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils.session import get_df, get_raw_df, is_data_loaded
from utils.llm import call_llm, get_active_model


# ============================================================================
# MODEL VISUALIZATION FUNCTIONS
# ============================================================================

def create_model_comparison_chart(metrics_dict: dict, task_type: str):
    """Create bar chart comparing model performance metrics."""
    
    if not metrics_dict:
        return None
    
    comparison_data = []
    
    for model_name, metrics in metrics_dict.items():
        row = {"Model": model_name}
        
        if task_type == "classification":
            metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
        else:
            metric_keys = ["R2 Score", "MAE", "RMSE"]
        
        for key in metric_keys:
            if key in metrics and isinstance(metrics[key], (int, float)):
                row[key] = round(metrics[key], 4)
        
        if "cv_mean" in metrics and metrics["cv_mean"]:
            row["CV Score"] = round(metrics["cv_mean"], 4)
            
        comparison_data.append(row)
    
    if not comparison_data:
        return None
        
    df_comparison = pd.DataFrame(comparison_data)
    
    if task_type == "classification":
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
        title = "📊 Model Performance Comparison"
        y_title = "Score (higher is better)"
    else:
        metrics_to_plot = ["R2 Score"]
        title = "📊 Model Performance Comparison"
        y_title = "R² Score (higher is better)"
    
    available_metrics = [m for m in metrics_to_plot if m in df_comparison.columns]
    
    if not available_metrics:
        return None
    
    fig = px.bar(
        df_comparison,
        x="Model",
        y=available_metrics,
        barmode="group",
        title=title,
        labels={"value": y_title, "variable": "Metric", "Model": "Algorithm"},
        color_discrete_sequence=px.colors.sequential.Viridis,
        text_auto='.3f'
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0.2)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e2f0")
    )
    
    fig.update_traces(
        textposition="outside",
        marker_line_width=1,
        marker_line_color="white"
    )
    
    return fig


def create_radar_chart(metrics_dict: dict, best_model: str):
    """Create radar chart comparing top models across multiple metrics."""
    
    if not metrics_dict:
        return None
    
    models = list(metrics_dict.keys())[:4]
    if best_model not in models and best_model in metrics_dict:
        models[0] = best_model
    
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "cv_mean"]
    
    fig = go.Figure()
    
    for model in models:
        if model not in metrics_dict:
            continue
            
        values = []
        for metric in metrics_list:
            val = metrics_dict[model].get(metric, 0)
            if isinstance(val, (int, float)):
                values.append(round(val, 3))
            else:
                values.append(0)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_list,
            fill='toself',
            name=model,
            line=dict(width=2),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color="#e2e2f0")
            ),
            angularaxis=dict(
                tickfont=dict(color="#e2e2f0", size=10)
            )
        ),
        template="plotly_dark",
        height=450,
        title="Multi-Metric Radar Comparison",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def create_feature_importance_chart(feature_importances, features_list, top_n: int = 10):
    """Create horizontal bar chart for feature importance."""
    
    if not feature_importances:
        return None
    
    # Handle both dict and list formats
    if isinstance(feature_importances, list):
        importances_dict = dict(zip(features_list, feature_importances))
    else:
        importances_dict = feature_importances
    
    sorted_features = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    df_importance = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])
    
    fig = px.bar(
        df_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances",
        labels={"Importance": "Importance Score", "Feature": ""},
        color="Importance",
        color_continuous_scale="Viridis",
        text_auto='.3f'
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        yaxis=dict(categoryorder="total ascending"),
        plot_bgcolor="rgba(0,0,0,0.2)",
        coloraxis_showscale=False
    )
    
    fig.update_traces(marker_line_width=0)
    
    return fig


def create_confusion_matrix_heatmap(cm_matrix):
    """Create confusion matrix heatmap."""
    
    if not cm_matrix:
        return None
    
    fig = px.imshow(
        cm_matrix,
        text_auto=True,
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        color_continuous_scale="Blues",
        aspect="auto"
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=450,
        plot_bgcolor="rgba(0,0,0,0.2)"
    )
    
    return fig


def generate_model_recommendation(metrics_dict: dict, task_type: str, dataset_size: int, cv_folds: int = 5):
    """Generate intelligent model recommendation based on performance."""
    
    if not metrics_dict:
        return None
    
    primary_metric = "Accuracy" if task_type == "classification" else "R2 Score"
    
    best_model = None
    best_score = -1
    
    for model, metrics in metrics_dict.items():
        score = metrics.get(primary_metric, -1)
        if isinstance(score, (int, float)) and score > best_score:
            best_score = score
            best_model = model
    
    if not best_model:
        return None
    
    best_metrics = metrics_dict[best_model]
    
    reasoning = []
    
    if task_type == "classification":
        accuracy = best_metrics.get("Accuracy", 0)
        f1 = best_metrics.get("F1-Score", 0)
        
        if accuracy >= 0.9:
            reasoning.append(f"✅ **Excellent predictive power** ({accuracy*100:.1f}% accuracy)")
        elif accuracy >= 0.8:
            reasoning.append(f"👍 **Strong predictive power** ({accuracy*100:.1f}% accuracy)")
        elif accuracy >= 0.7:
            reasoning.append(f"📊 **Moderate predictive power** ({accuracy*100:.1f}% accuracy)")
        else:
            reasoning.append(f"⚠️ **Room for improvement** ({accuracy*100:.1f}% accuracy)")
        
        precision = best_metrics.get("Precision", 0)
        recall = best_metrics.get("Recall", 0)
        if abs(precision - recall) < 0.05:
            reasoning.append("⚖️ **Well-balanced** between precision and recall")
        else:
            reasoning.append(f"⚖️ **Trade-off detected**: Precision={precision:.3f}, Recall={recall:.3f}")
        
        if f1 >= 0.85:
            reasoning.append("🎯 **Excellent F1-score** - great overall performance")
        elif f1 >= 0.7:
            reasoning.append("🎯 **Good F1-score** - reliable predictions")
            
    else:
        r2 = best_metrics.get("R2 Score", 0)
        rmse = best_metrics.get("RMSE", 0)
        
        if r2 >= 0.9:
            reasoning.append(f"✅ **Excellent fit** (R² = {r2:.3f})")
        elif r2 >= 0.7:
            reasoning.append(f"👍 **Strong fit** (R² = {r2:.3f})")
        elif r2 >= 0.5:
            reasoning.append(f"📊 **Moderate fit** (R² = {r2:.3f})")
        else:
            reasoning.append(f"⚠️ **Weak fit** (R² = {r2:.3f}) - consider feature engineering")
        
        reasoning.append(f"📏 **Prediction error**: RMSE = {rmse:.4f}")
    
    cv_mean = best_metrics.get("cv_mean", None)
    cv_std = best_metrics.get("cv_std", None)
    
    if cv_mean and cv_std:
        if cv_std < 0.05:
            reasoning.append(f"🔄 **Highly stable** across {cv_folds}-fold CV (std = {cv_std:.4f})")
        elif cv_std < 0.1:
            reasoning.append(f"🔄 **Moderately stable** across CV folds (std = {cv_std:.4f})")
        else:
            reasoning.append(f"🔄 **High variance** detected - consider more data or regularization")
    
    if dataset_size < 1000:
        reasoning.append("💡 **Small dataset detected** - consider simpler models or data augmentation")
    elif dataset_size > 100000:
        reasoning.append("🚀 **Large dataset** - gradient boosting or deep learning could improve results")
    
    alternatives = []
    for model, metrics in metrics_dict.items():
        if model != best_model:
            score = metrics.get(primary_metric, 0)
            if isinstance(score, (int, float)):
                diff = (best_score - score) * 100
                if diff < 5:
                    alternatives.append(f"{model} (within {diff:.1f}% of best)")
    
    recommendation = {
        "best_model": best_model,
        "best_score": best_score,
        "reasoning": reasoning,
        "alternatives": alternatives[:3],
        "primary_metric": primary_metric
    }
    
    return recommendation


def build_comparison_dataframe(metrics_dict: dict, task_type: str) -> pd.DataFrame:
    """Build comparison dataframe for metrics display."""
    
    data = []
    for model_name, metrics in metrics_dict.items():
        row = {"Model": model_name}
        
        if task_type == "classification":
            for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                val = metrics.get(metric, None)
                if isinstance(val, (int, float)):
                    row[metric] = round(val, 4)
        else:
            for metric in ["R2 Score", "MAE", "RMSE"]:
                val = metrics.get(metric, None)
                if isinstance(val, (int, float)):
                    row[metric] = round(val, 4)
        
        if "cv_mean" in metrics and metrics["cv_mean"]:
            row["CV Score"] = round(metrics["cv_mean"], 4)
        
        data.append(row)
    
    return pd.DataFrame(data)


# ============================================================================
# REPORT RENDERING FUNCTIONS
# ============================================================================

def render_quality_section():
    """Render data quality and cleaning insights section."""
    
    quality_report = st.session_state.get("quality_report")
    cleaning_report = st.session_state.get("cleaning_report")
    
    if not quality_report:
        st.info("📊 Run the Data Quality Agent first to see quality insights.")
        return
    
    st.markdown("## 🔍 Data Quality & Cleaning Insights")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = quality_report.get('score', 0)
        color = "#34d399" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
            <div style="font-size: 2rem; color: {color}; font-weight: bold;">{score:.0f}</div>
            <div style="font-size: 0.8rem; color: #a0a0c0;">Quality Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        completeness = quality_report.get('completeness', 0)
        st.metric("Completeness", f"{completeness:.1f}%")
    
    with col3:
        duplicate_rate = quality_report.get('duplicate_rate', 0)
        st.metric("Duplicate Rate", f"{duplicate_rate:.1f}%")
    
    with col4:
        missing_cells = quality_report.get('missing_cells', 0)
        st.metric("Missing Cells", f"{missing_cells:,}")
    
    if cleaning_report:
        st.markdown("### 🧹 Cleaning Operations Summary")
        
        before_shape = cleaning_report.get('before_shape', [0, 0])
        after_shape = cleaning_report.get('after_shape', [0, 0])
        before_rows = before_shape[0] if before_shape else 0
        after_rows = after_shape[0] if after_shape else 0
        rows_removed = before_rows - after_rows
        removal_pct = (rows_removed / before_rows * 100) if before_rows > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows Before", f"{before_rows:,}")
            st.metric("Rows After", f"{after_rows:,}")
        with col2:
            st.metric("Rows Removed", f"{rows_removed:,}", delta=f"-{removal_pct:.1f}%")
        
        cleaning_log = cleaning_report.get('log', [])
        if cleaning_log:
            with st.expander("📋 View Cleaning Log"):
                for log_entry in cleaning_log:
                    st.info(log_entry)
    
    st.markdown("### 💡 Quality Insights")
    
    insights = []
    
    if quality_report.get('completeness', 100) < 90:
        insights.append("⚠️ **Missing values detected** - Consider investigating data collection processes")
    
    if quality_report.get('duplicate_rate', 0) > 5:
        insights.append("🔄 **High duplicate rate** - Review data entry or merging processes")
    
    score_val = quality_report.get('score', 100)
    if score_val >= 85:
        insights.append("✅ **Excellent data quality** - Dataset is ready for advanced analysis")
    elif score_val >= 70:
        insights.append("👍 **Good data quality** - Minor issues addressed by cleaning")
    else:
        insights.append("⚠️ **Data quality needs attention** - Consider additional validation steps")
    
    for insight in insights:
        st.markdown(insight)


def render_statistics_summary():
    """Render quick statistical summary."""
    st.markdown("## 📊 Statistical Summary")
    st.markdown("---")
    
    df = get_df()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    if numeric_cols:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(numeric_cols))
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            max_corr = 0
            max_pair = ("", "")
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > max_corr and corr_val < 1:
                        max_corr = corr_val
                        max_pair = (numeric_cols[i], numeric_cols[j])
            
            if max_corr > 0:
                st.info(f"🔗 **Strongest Correlation:** `{max_pair[0]}` ↔ `{max_pair[1]}` ({max_corr:.3f})")
        
        # Show basic stats for first few numeric columns
        with st.expander("📈 View Numeric Statistics"):
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
    else:
        st.info("No numeric columns found for statistical analysis")


def render_modeling_section():
    """Render dynamic modeling results section with visualizations."""
    
    if not st.session_state.get("modeling_done"):
        st.info("🤖 Run the Modeling Agent first to see model comparison charts and recommendations!")
        return
    
    st.markdown("## 🤖 Machine Learning Modeling Results")
    st.markdown("---")
    
    metrics_dict = st.session_state.model_metrics
    task_type = st.session_state.model_task_type
    best_model_name = st.session_state.trained_model_name
    features_list = st.session_state.model_features_list or []
    target_col = st.session_state.model_target_col
    dataset_size = st.session_state.df_clean.shape[0] if st.session_state.df_clean is not None else 0
    
    if not metrics_dict:
        st.warning("No modeling metrics available.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Comparison", 
        "🎯 Best Model Details", 
        "🔑 Feature Importance",
        "📈 Recommendations"
    ])
    
    with tab1:
        st.markdown("### Model Performance Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_bar = create_model_comparison_chart(metrics_dict, task_type)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No comparable metrics available")
        
        with col2:
            st.markdown("#### 📋 Metrics Table")
            comparison_df = build_comparison_dataframe(metrics_dict, task_type)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        fig_radar = create_radar_chart(metrics_dict, best_model_name)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)
        
        if task_type == "classification":
            best_metrics = metrics_dict.get(best_model_name, {})
            if "confusion_matrix" in best_metrics and best_metrics["confusion_matrix"]:
                st.markdown("#### Confusion Matrix (Best Model)")
                fig_cm = create_confusion_matrix_heatmap(best_metrics["confusion_matrix"])
                if fig_cm:
                    st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab2:
        st.markdown(f"### 🏆 Best Model: **{best_model_name}**")
        
        best_metrics = metrics_dict[best_model_name]
        
        if task_type == "classification":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{best_metrics.get('Accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{best_metrics.get('Precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{best_metrics.get('Recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{best_metrics.get('F1-Score', 0):.4f}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{best_metrics.get('R2 Score', 0):.4f}")
            with col2:
                st.metric("MAE", f"{best_metrics.get('MAE', 0):.4f}")
            with col3:
                st.metric("RMSE", f"{best_metrics.get('RMSE', 0):.4f}")
        
        if "cv_mean" in best_metrics and best_metrics["cv_mean"]:
            cv_text = f"{best_metrics['cv_mean']:.4f} ± {best_metrics.get('cv_std', 0):.4f}"
            st.metric("Cross-Validation Score", cv_text)
        
        st.markdown("#### ⚙️ Model Configuration")
        test_size = st.session_state.get('test_size', 0.2)
        st.markdown(f"""
        - **Task Type:** `{task_type.upper()}`
        - **Target Variable:** `{target_col}`
        - **Features Used:** {len(features_list)} columns
        - **Training/Test Split:** {int((1 - test_size) * 100)}% / {int(test_size * 100)}%
        """)
    
    with tab3:
        st.markdown("### 🔑 Feature Importance Analysis")
        
        best_metrics = metrics_dict.get(best_model_name, {})
        
        if "feature_importances" in best_metrics:
            fig_importance = create_feature_importance_chart(
                best_metrics["feature_importances"], 
                features_list, 
                top_n=10
            )
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with st.expander("📋 View All Feature Importances"):
                if isinstance(best_metrics["feature_importances"], list):
                    importance_df = pd.DataFrame({
                        "Feature": features_list,
                        "Importance": best_metrics["feature_importances"]
                    }).sort_values("Importance", ascending=False)
                else:
                    importance_df = pd.DataFrame([
                        {"Feature": feat, "Importance": imp}
                        for feat, imp in best_metrics["feature_importances"].items()
                    ]).sort_values("Importance", ascending=False)
                
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
            
            # Key insight about top features
            if isinstance(best_metrics["feature_importances"], list):
                top_features = sorted(zip(features_list, best_metrics["feature_importances"]), 
                                    key=lambda x: x[1], reverse=True)[:3]
            else:
                top_features = sorted(best_metrics["feature_importances"].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
            
            if top_features:
                top_names = [f"`{f[0]}`" for f in top_features]
                st.info(f"🎯 The top 3 most influential features are: {', '.join(top_names)}. "
                       f"These drive the majority of predictive power in the {best_model_name} model.")
        else:
            st.info("Feature importance not available for this model type")
    
    with tab4:
        st.markdown("### 📊 Model Selection Recommendations")
        
        cv_folds = st.session_state.get('cv_folds', 5)
        recommendation = generate_model_recommendation(metrics_dict, task_type, dataset_size, cv_folds)
        
        if recommendation:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
                        border: 2px solid #6366f1;
                        border-radius: 16px;
                        padding: 1.5rem;
                        margin: 1rem 0;">
                <h4 style="color: #a78bfa; margin-bottom: 0.5rem;">🏆 Recommended Model: {recommendation['best_model']}</h4>
                <p style="font-size: 1.2rem; font-weight: bold; color: #34d399;">
                    {recommendation['primary_metric']}: {recommendation['best_score']:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🧠 Why This Model?")
            for reason in recommendation["reasoning"]:
                st.markdown(reason)
            
            if recommendation["alternatives"]:
                st.markdown("#### 🔄 Alternative Models to Consider")
                st.markdown(f"*{', '.join(recommendation['alternatives'])}*")
            
            st.markdown("#### 🚀 Actionable Next Steps")
            
            next_steps = []
            best_score = recommendation['best_score']
            
            if best_score < 0.8 and task_type == "classification":
                next_steps.append("📊 **Feature Engineering**: Create interaction features or polynomial features")
                next_steps.append("🔧 **Hyperparameter Tuning**: Use GridSearchCV or Optuna for optimization")
                next_steps.append("📈 **Ensemble Methods**: Try stacking or voting classifiers")
            elif best_score < 0.85 and task_type == "regression":
                next_steps.append("🔍 **Outlier Investigation**: Review extreme values affecting predictions")
                next_steps.append("➕ **Additional Features**: Consider external data sources")
                next_steps.append("🎛️ **Model Complexity**: Try XGBoost or LightGBM with tuning")
            elif best_score >= 0.85:
                next_steps.append("✅ **Model Deployment**: Ready for production deployment")
                next_steps.append("📝 **Model Documentation**: Document feature requirements and preprocessing")
                next_steps.append("📊 **Monitoring Setup**: Implement drift detection for production")
            
            for step in next_steps[:3]:
                st.markdown(step)
        else:
            st.warning("Unable to generate recommendations - insufficient metrics data")


def show_pipeline_status():
    """Show which agents have completed."""
    
    steps = [
        ("🔍 Quality", st.session_state.get("quality_report") is not None),
        ("🧹 Cleaning", st.session_state.get("cleaning_report") is not None),
        ("📊 Stats", st.session_state.get("stats_done", False)),
        ("📈 Viz", st.session_state.get("viz_done", False)),
        ("🤖 Modeling", st.session_state.get("modeling_done", False)),
        ("💡 Insights", st.session_state.get("insights_text") is not None),
    ]
    
    st.markdown("### 📋 Pipeline Status")
    
    cols = st.columns(len(steps))
    for col, (label, done) in zip(cols, steps):
        icon = "✅" if done else "⏳"
        color = "#34d399" if done else "#64748b"
        col.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:1.5rem;">{icon}</div>
            <div style="font-size:0.7rem; color:{color};">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    completed = sum(1 for _, done in steps if done)
    st.progress(completed / len(steps))
    st.caption(f"Pipeline Progress: {completed}/{len(steps)} agents complete")


def format_report_text(text: str) -> str:
    """Format report text with HTML styling."""
    
    if not text:
        return "<p>No report content available.</p>"
    
    text = re.sub(r'^# (.*?)$', r'<h1 style="color: #f093fb; font-size: 2rem; margin-top: 0; margin-bottom: 0.5rem;">\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2 style="color: #a78bfa; font-size: 1.5rem; margin-top: 1.5rem; margin-bottom: 0.75rem; border-left: 4px solid #a78bfa; padding-left: 0.8rem;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3 style="color: #c4b5fd; font-size: 1.2rem; margin-top: 1rem; margin-bottom: 0.5rem;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #f093fb;">\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em style="color: rgba(255,255,255,0.8);">\1</em>', text)
    
    lines = text.split('\n')
    formatted_lines = []
    in_table = False
    table_html = ""
    
    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_html = '<div style="overflow-x: auto; margin: 1.2rem 0;"><table style="width: 100%; border-collapse: collapse; background: rgba(0,0,0,0.2); border-radius: 12px; overflow: hidden;">'
            
            if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                continue
            
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            table_html += '<tr>'
            for cell in cells:
                cell_clean = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                table_html += f'<td style="border: 1px solid rgba(255,255,255,0.15); padding: 0.6rem; color: rgba(255,255,255,0.85);">{cell_clean}</td>'
            table_html += '</tr>'
        else:
            if in_table:
                table_html += '</table></div>'
                formatted_lines.append(table_html)
                in_table = False
                table_html = ""
            formatted_lines.append(line)
    
    if in_table:
        table_html += '</table></div>'
        formatted_lines.append(table_html)
    
    text = '\n'.join(formatted_lines)
    text = re.sub(r'^[\-\*]\s+(.*?)$', r'<li style="margin-bottom: 0.5rem;">• \1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li.*?>.*?</li>\n?)+', r'<ul style="margin: 0.8rem 0; padding-left: 1.5rem;">\g<0></ul>', text, flags=re.DOTALL)
    text = text.replace('\n\n', '<br><br>')
    
    return text


def render_export_options(df: pd.DataFrame):
    """Render export options for report."""
    
    st.markdown("## 📤 Export Options")
    st.markdown("Download the cleaned data and report in your preferred format:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📄 CSV",
            data=csv_data,
            file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with col2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Report_Data", index=False)
        excel_data = output.getvalue()
        st.download_button(
            "📊 Excel",
            data=excel_data,
            file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    
    with col3:
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            "📋 JSON",
            data=json_data,
            file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    
    with col4:
        html_data = df.to_html(index=False, border=0, classes='dataframe')
        st.download_button(
            "🌐 HTML",
            data=html_data,
            file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True,
        )
    
    st.markdown("### 📝 Report Downloads")
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        if st.session_state.get("full_report_md"):
            st.download_button(
                "📄 Download Report (Markdown)",
                data=st.session_state.full_report_md,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
    
    with col_rep2:
        if st.session_state.get("report_text"):
            st.download_button(
                "📝 Download Report (Text)",
                data=st.session_state.report_text,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )


def generate_full_markdown_report(df: pd.DataFrame, llm_report: str, file_name: str) -> str:
    """Generate complete markdown report."""
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    quality_report = st.session_state.get("quality_report", {})
    cleaning_report = st.session_state.get("cleaning_report", {})
    insights_text = st.session_state.get("insights_text", "Not generated.")
    
    lines = []
    lines.append(f"# 📊 Data Analysis Report")
    lines.append(f"")
    lines.append(f"**File:** `{file_name}`  ")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Dataset Size:** {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    lines.append("---\n")
    
    # Dataset Overview
    lines.append("## 📋 1. Dataset Overview\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Rows | **{df.shape[0]:,}** |")
    lines.append(f"| Total Columns | **{df.shape[1]}** |")
    lines.append(f"| Numeric Columns | **{len(numeric_cols)}** |")
    lines.append(f"| Categorical Columns | **{len(cat_cols)}** |")
    lines.append(f"| Missing Cells | **{df.isnull().sum().sum():,}** |\n")
    
    # Quality Section
    if quality_report:
        lines.append("## ✅ 2. Data Quality Assessment\n")
        score = quality_report.get('score', 0)
        lines.append(f"### Overall Quality Score: **{score}/100**\n")
        lines.append("| Dimension | Score |")
        lines.append("|-----------|-------|")
        lines.append(f"| Completeness | {quality_report.get('completeness', 0)}% |")
        lines.append(f"| Duplicate Rate | {quality_report.get('duplicate_rate', 0)}% |\n")
    
    # Cleaning Section
    if cleaning_report:
        lines.append("## 🧹 3. Data Cleaning Summary\n")
        before_shape = cleaning_report.get('before_shape', [0, 0])
        after_shape = cleaning_report.get('after_shape', [0, 0])
        lines.append(f"- **Rows before cleaning:** {before_shape[0]:,}")
        lines.append(f"- **Rows after cleaning:** {after_shape[0]:,}")
        lines.append(f"- **Rows removed:** {before_shape[0] - after_shape[0]:,}\n")
        
        if cleaning_report.get('log'):
            lines.append("### Cleaning Operations Performed:\n")
            for log_entry in cleaning_report['log']:
                lines.append(f"- {log_entry}")
            lines.append("")
    
    # Modeling Section
    if st.session_state.get("modeling_done"):
        offset = 4 if cleaning_report else 3
        lines.append(f"## 🤖 {offset}. Machine Learning Modeling\n")
        lines.append(f"**Target Variable:** `{st.session_state.model_target_col}`  ")
        lines.append(f"**Task Type:** `{st.session_state.model_task_type}`  ")
        lines.append(f"**Best Algorithm:** `{st.session_state.trained_model_name}`\n")
        
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        metrics = st.session_state.model_metrics[st.session_state.trained_model_name]
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k not in ["confusion_matrix"]:
                lines.append(f"| {k} | {v:.4f} |")
        lines.append("")
        
        if "feature_importances" in metrics:
            lines.append("### Feature Importances\n")
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            if isinstance(metrics["feature_importances"], list):
                feat_imp = dict(zip(st.session_state.model_features_list, metrics["feature_importances"]))
            else:
                feat_imp = metrics["feature_importances"]
            sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
            for feat, val in sorted_imp[:10]:
                lines.append(f"| {feat} | {val:.4f} |")
            lines.append("")
    
    # AI Insights
    insight_offset = 5 if st.session_state.get("modeling_done") else (4 if cleaning_report else 3)
    lines.append(f"## 💡 {insight_offset}. AI-Generated Insights\n")
    lines.append(insights_text)
    lines.append("\n")
    
    lines.append("---")
    lines.append(f"*Report generated by CSV Insight Agents · {now}*")
    
    return "\n".join(lines)


# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render():
    """Main render function for the Report page."""
    
    st.markdown("""
    <style>
    .report-header {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(102,126,234,0.3);
    }
    .status-complete {
        background: linear-gradient(135deg, rgba(0,176,155,0.2), rgba(150,201,61,0.2));
        border-left: 4px solid #34d399;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📄 Report Writing Agent")
    st.markdown("Compiles all agent outputs into a comprehensive, downloadable analysis report with visualizations.")
    st.markdown("---")
    
    if not is_data_loaded():
        st.warning("⚠️ No data loaded. Please upload a CSV on the Home page first.")
        if st.button("← Back to Home", use_container_width=False):
            st.session_state.current_page = "home"
            st.rerun()
        return
    
    df = get_df()
    file_name = st.session_state.get("file_name", "data.csv")
    
    # Show pipeline status
    show_pipeline_status()
    st.markdown("---")
    
    # Report Configuration
    st.markdown("### ⚙️ Report Configuration")
    
    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        report_tone = st.selectbox(
            "🎨 Report Tone",
            ["Executive (non-technical)", "Technical (data science)", "Mixed (balanced)"],
            help="Choose the writing style that best suits your audience"
        )
    with col_opts2:
        include_recs = st.checkbox("💡 Include Recommendations Section", value=True)
    
    # Generate Report Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_btn = st.button("🚀 Generate Full Report", use_container_width=True, type="primary")
    
    if generate_btn:
        try:
            # Build context for LLM
            context = build_full_context(df)
            
            system_prompt = f"""You are a senior data analyst writing a professional analysis report.
Tone: {report_tone}.
Write clearly, use structure (headings, bullet points), and be specific with numbers.
Use markdown formatting for better readability."""
            
            user_prompt = f"""Write a comprehensive data analysis report based on the following dataset analysis.

{context}

Structure your report with these sections:

## Executive Summary
- 3-4 sentences overview of key findings

## Key Findings  
- 5 bullet points with specific numbers and **bold** for important metrics

## Data Quality Assessment
- Brief assessment of data completeness and issues

## Trend Analysis
- 3-4 patterns observed in the data with specific examples

{f'## Strategic Recommendations' if include_recs else ''}
{f'- 3 actionable, specific next steps' if include_recs else ''}

Keep the report professional, specific, and under 600 words. Use markdown formatting."""
            
            with st.spinner("📝 Generating comprehensive report..."):
                llm_report = call_llm(user_prompt, system_prompt, max_tokens=1600)
                st.session_state.report_text = llm_report
                
                full_report = generate_full_markdown_report(df, llm_report, file_name)
                st.session_state.full_report_md = full_report
            
            st.success("✅ Report generated successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
    
    # Display Report Sections (Dynamic based on completed agents)
    st.markdown("---")
    
    # Always show quality section if available
    render_quality_section()
    
    # Show statistics if done
    if st.session_state.get("stats_done"):
        st.markdown("---")
        render_statistics_summary()
    
    # Show modeling section if done
    if st.session_state.get("modeling_done"):
        st.markdown("---")
        render_modeling_section()
    
    # Show AI Insights if available
    if st.session_state.get("insights_text"):
        st.markdown("---")
        st.markdown("## 💡 AI-Generated Business Insights")
        formatted_insights = format_report_text(st.session_state.insights_text)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(102,126,234,0.3);
                    border-radius: 20px;
                    padding: 2rem;
                    line-height: 1.7;">
            {formatted_insights}
        </div>
        """, unsafe_allow_html=True)
    
    # Export Options
    if is_data_loaded():
        st.markdown("---")
        render_export_options(df)
    
    # Navigation
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    with col_nav1:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
    with col_nav2:
        if st.button("💡 Back to Insights", use_container_width=True):
            st.session_state.current_page = "insights"
            st.rerun()
    with col_nav3:
        if st.button("🔄 New Analysis", use_container_width=True):
            # Reset relevant session states
            for key in ["quality_report", "cleaning_report", "stats_done", "viz_done", 
                       "modeling_done", "insights_text", "report_text"]:
                if key in st.session_state:
                    st.session_state[key] = None
            st.session_state.current_page = "home"
            st.rerun()


def build_full_context(df: pd.DataFrame) -> str:
    """Build comprehensive context for the LLM report."""
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    parts = []
    parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    parts.append(f"Columns: {', '.join(df.columns.tolist()[:15])}")
    
    if len(df.columns) > 15:
        parts.append(f"... and {len(df.columns) - 15} more columns")
    
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        parts.append("\nNumeric Statistics:\n" + desc.to_string())
    
    if cat_cols:
        parts.append("\nCategorical Columns:")
        for col in cat_cols[:4]:
            top = df[col].value_counts().head(3)
            parts.append(f"  {col}: {dict(top)}")
    
    quality_report = st.session_state.get("quality_report")
    if quality_report and isinstance(quality_report, dict):
        score = quality_report.get('score', 'N/A')
        completeness = quality_report.get('completeness', 'N/A')
        parts.append(f"\nData Quality Score: {score}/100")
        parts.append(f"Completeness: {completeness}%")
    
    if st.session_state.get("modeling_done"):
        parts.append(f"\nMachine Learning Model: {st.session_state.trained_model_name}")
        parts.append(f"Predicting Target: {st.session_state.model_target_col}")
        parts.append(f"Task type: {st.session_state.model_task_type}")
        metrics = st.session_state.model_metrics[st.session_state.trained_model_name]
        parts.append("Model Performance Metrics:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                parts.append(f"  {k}: {v:.4f}")
    
    insights = st.session_state.get("insights_text")
    if insights:
        parts.append(f"\nAI Insights Summary:\n{insights[:500]}")
    
    cleaning_report = st.session_state.get("cleaning_report")
    if cleaning_report and isinstance(cleaning_report, dict):
        before = cleaning_report.get('before_shape', [0, 0])[0]
        after = cleaning_report.get('after_shape', [0, 0])[0]
        parts.append(f"\nCleaning Summary:")
        parts.append(f"  - Rows before: {before}")
        parts.append(f"  - Rows after: {after}")
        parts.append(f"  - Rows removed: {before - after}")
    
    return "\n".join(parts)
