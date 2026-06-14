import streamlit as st
import pandas as pd
import numpy as np
import re
from utils.session import get_df, is_data_loaded
from utils.llm import call_llm, get_active_model


def build_data_summary(df: pd.DataFrame) -> str:
    """Build a compact text summary of the dataframe to send to LLM."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    lines = []
    lines.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}")
    lines.append(f"Categorical columns: {', '.join(cat_cols) if cat_cols else 'None'}")
    lines.append(f"Missing values: {df.isnull().sum().sum()} total cells")
    lines.append("")

    if numeric_cols:
        lines.append("=== Numeric Summary ===")
        desc = df[numeric_cols].describe().round(3)
        lines.append(desc.to_string())
        lines.append("")

    if cat_cols:
        lines.append("=== Categorical Summary ===")
        for col in cat_cols[:5]:
            top = df[col].value_counts().head(5)
            lines.append(f"{col}: {dict(top)}")
        lines.append("")

    if len(numeric_cols) >= 2:
        lines.append("=== Top Correlations ===")
        corr = df[numeric_cols].corr()
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], round(corr.iloc[i, j], 3)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for a, b, val in pairs[:5]:
            lines.append(f"  {a} <-> {b}: {val}")

    # Add modeling info if available
    if st.session_state.get("modeling_done"):
        lines.append("")
        lines.append("=== Machine Learning Modeling Results ===")
        lines.append(f"Target Column: {st.session_state.model_target_col}")
        lines.append(f"Features Used: {', '.join(st.session_state.model_features_list)}")
        lines.append(f"Task Type: {st.session_state.model_task_type}")
        lines.append(f"Best Performing Model: {st.session_state.trained_model_name}")
        lines.append("Model Performance Metrics (Test Set):")
        metrics = st.session_state.model_metrics[st.session_state.trained_model_name]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                lines.append(f"  {k}: {v:.4f}")
        
        # Feature importances
        if "feature_importances" in metrics:
            lines.append("Top Feature Importances (Most Predictive):")
            feat_imp = metrics["feature_importances"]
            features = st.session_state.model_features_list
            if len(feat_imp) == len(features):
                sorted_imp = sorted(zip(features, feat_imp), key=lambda x: x[1], reverse=True)
                for feat, val in sorted_imp[:5]:
                    lines.append(f"  {feat}: {val:.4f}")

    return "\n".join(lines)


def format_ai_response(text: str) -> str:
    """Format AI response with better HTML styling."""
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"
    
    text = re.sub(r'```.*?```', save_code_block, text, flags=re.DOTALL)
    
    text = re.sub(r'### (.*?)\n', r'<h3 style="color: #10B981; margin-top: 1rem; margin-bottom: 0.5rem;">\1</h3>\n', text)
    text = re.sub(r'## (.*?)\n', r'<h2 style="color: #3B82F6; margin-top: 1.25rem; margin-bottom: 0.75rem; border-left: 3px solid #3B82F6; padding-left: 0.75rem;">\1</h2>\n', text)
    
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #10B981;">\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em style="color: var(--text-secondary);">\1</em>', text)
    
    text = re.sub(r'^[\-\*]\s+(.*?)$', r'<li style="margin-bottom: 0.25rem;">• \1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^[0-9]+\.\s+(.*?)$', r'<li style="margin-bottom: 0.25rem;"><strong style="color: #3B82F6;">✓</strong> \1</li>', text, flags=re.MULTILINE)
    
    text = re.sub(r'(<li.*?>.*?</li>\n?)+', r'<ul style="margin: 0.5rem 0; padding-left: 1.25rem;">\g<0></ul>', text, flags=re.DOTALL)
    text = re.sub(r'^---$', r'<hr style="margin: 1rem 0; border-color: var(--border-color);">', text, flags=re.MULTILINE)
    
    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)
    
    return text


def render():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: var(--text-primary); margin-bottom: 0.25rem;">💡 AI Insights Agent</h2>
        <p style="color: var(--text-secondary);">Generate business-level insights from your data using LLM</p>
    </div>
    """, unsafe_allow_html=True)

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        if st.button("← Back to Home"):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()

    # Model info
    model_name = get_active_model()
    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 1rem;">
        <span style="color: var(--text-secondary);">🤖 Active Model: </span>
        <code style="color: #10B981; background: var(--bg-secondary); padding: 0.25rem 0.5rem; border-radius: 6px;">{model_name}</code>
        <span style="color: #10B981; font-size: 0.75rem; margin-left: 0.5rem;">✓ OpenRouter</span>
    </div>
    """, unsafe_allow_html=True)

    # Insight type selector
    insight_options = {
        "General business insights": "📊",
        "Anomalies and red flags": "⚠️",
        "Growth opportunities": "🚀",
        "Customer behavior patterns": "👥",
        "Operational efficiency observations": "⚙️",
        "Executive summary (non-technical)": "📋"
    }
    
    insight_type = st.selectbox(
        "Select insight type",
        list(insight_options.keys()),
        format_func=lambda x: f"{insight_options[x]} {x}"
    )

    extra_context = st.text_area(
        "Additional context (optional)",
        placeholder="Example: This is monthly sales data from an e-commerce company targeting Southeast Asia...",
        height=80,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Generate Insights", use_container_width=True, type="primary")

    if run_btn:
        data_summary = build_data_summary(df)

        # Add modeling results to context if available
        modeling_context = ""
        if st.session_state.get("modeling_done"):
            modeling_context = f"""
            \n\nNote: A machine learning model has been trained on this data.
            The best performing model was {st.session_state.trained_model_name} predicting '{st.session_state.model_target_col}'.
            Key predictive features include: {', '.join(st.session_state.model_features_list[:5])}.
            Consider these insights when making recommendations.
            """

        system_prompt = """You are a senior data analyst and business intelligence expert. 
You analyze datasets and provide clear, actionable, business-focused insights.
Write in a professional but accessible tone. Use bullet points and short paragraphs.
Always be specific and reference actual column names and values from the data.
Format your response with clear sections using markdown.
Use **bold** for key metrics and important numbers.
Avoid generic statements. Every insight must be grounded in the data provided."""

        user_prompt = f"""Analyze the following dataset and provide {insight_type.lower()}.

{extra_context if extra_context.strip() else ''}{modeling_context}

=== DATASET SUMMARY ===
{data_summary}

Please provide a well-structured response with:

## Key Findings
- 3-5 bullet points with specific numbers and **bold** for important metrics

## Patterns & Trends
- What stands out in the data

## Actionable Recommendations
- 2-3 concrete next steps with clear justifications

## Data Quality Notes
- Any concerns worth flagging

Keep the total response under 600 words. Be specific and data-driven."""

        with st.spinner(f"🧠 Analyzing with {model_name}..."):
            response = call_llm(user_prompt, system_prompt, max_tokens=1500)
            st.session_state.insights_text = response
            st.session_state.insights_type = insight_type

    # Display insights
    if st.session_state.get("insights_text"):
        st.markdown("---")
        
        insight_icon = insight_options.get(st.session_state.get('insights_type', ''), "💡")
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">{insight_icon}</span>
            <h2 style="display: inline-block; margin-left: 0.5rem;">{st.session_state.get('insights_type', 'AI Insights')}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        formatted_response = format_ai_response(st.session_state.insights_text)
        
        st.markdown(f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 1.5rem; line-height: 1.6;">
            {formatted_response}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button("📥 Download", data=st.session_state.insights_text, file_name=f"insights.txt", use_container_width=True)
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state.insights_text = None
                st.rerun()
        with col3:
            if st.button("📋 Copy", use_container_width=True):
                st.info("Select and copy the text above")
        with col4:
            if st.button("➡️ Report", use_container_width=True):
                st.session_state.current_page = "report"
                st.rerun()
        
        # Follow-up question section - FIXED: Added proper label
        st.markdown("---")
        st.markdown("### ❓ Ask a Follow-up Question")
        
        col_q1, col_q2 = st.columns([4, 1])
        with col_q1:
            custom_q = st.text_input(
                "Your question",  # Added proper label
                placeholder="Example: What are the top 3 actionable recommendations from this data?",
                label_visibility="collapsed",
                key="custom_question"
            )
        with col_q2:
            if st.button("Ask", key="custom_ask", use_container_width=True):
                if custom_q.strip():
                    data_summary = build_data_summary(df)
                    prompt = f"""Dataset summary:
{data_summary}

Question: {custom_q}

Answer the question in a clear, well-structured format. Use markdown for better readability. Keep the answer concise but informative (max 300 words)"""

                    with st.spinner("💭 Generating answer..."):
                        answer = call_llm(prompt, max_tokens=800)
                        
                        st.markdown(f"""
                        <div style="background: var(--bg-card); border-left: 4px solid #3B82F6; border-radius: 12px; padding: 1rem; margin-top: 0.75rem;">
                            <div style="color: var(--text-secondary); font-size: 0.8rem; margin-bottom: 0.5rem;">💬 Answer:</div>
                            <div style="color: var(--text-primary); line-height: 1.6;">{answer}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter a question.")
        
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("← Back to Modeling", use_container_width=True):
                st.session_state.current_page = "modeling"
                st.rerun()
        with col_nav2:
            if st.button("➡️ Proceed to Report", use_container_width=True):
                st.session_state.current_page = "report"
                st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧠</div>
            <div style="font-size: 1.2rem; font-weight: 600;">Ready to Generate Insights</div>
            <div style="color: var(--text-secondary);">Select an insight type and click "Generate Insights" to begin</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Back to Home"):
            st.session_state.current_page = "home"
            st.rerun()
