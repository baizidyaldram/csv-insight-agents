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

    # Add data quality info if available
    quality_report = st.session_state.get("quality_report")
    if quality_report and isinstance(quality_report, dict):
        lines.append("")
        lines.append("=== Data Quality Assessment ===")
        lines.append(f"Overall Quality Score: {quality_report.get('score', 'N/A')}/100")
        lines.append(f"Completeness: {quality_report.get('completeness', 'N/A')}%")
        lines.append(f"Duplicate Rate: {quality_report.get('duplicate_rate', 'N/A')}%")
        lines.append(f"Missing Cells: {quality_report.get('missing_cells', 'N/A')}")

    # Add cleaning info if available
    cleaning_report = st.session_state.get("cleaning_report")
    if cleaning_report and isinstance(cleaning_report, dict):
        lines.append("")
        lines.append("=== Data Cleaning Summary ===")
        before_shape = cleaning_report.get('before_shape', [0, 0])
        after_shape = cleaning_report.get('after_shape', [0, 0])
        lines.append(f"Rows before cleaning: {before_shape[0]}")
        lines.append(f"Rows after cleaning: {after_shape[0]}")
        lines.append(f"Rows removed: {before_shape[0] - after_shape[0]}")
        if cleaning_report.get('log'):
            lines.append(f"Cleaning operations: {len(cleaning_report['log'])} steps performed")

    # Add modeling info if available - FIXED to handle both list and dict
    if st.session_state.get("modeling_done"):
        lines.append("")
        lines.append("=== Machine Learning Modeling Summary ===")
        lines.append(f"Target Column: {st.session_state.model_target_col}")
        lines.append(f"Features Used: {', '.join(st.session_state.model_features_list[:10])}")
        if len(st.session_state.model_features_list) > 10:
            lines.append(f"  ... and {len(st.session_state.model_features_list) - 10} more features")
        lines.append(f"Task Type: {st.session_state.model_task_type}")
        lines.append(f"Best Trained Model: {st.session_state.trained_model_name}")
        lines.append("Model Performance Metrics on Test Set:")
        
        metrics = st.session_state.model_metrics.get(st.session_state.trained_model_name, {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k not in ["cv_std"]:
                lines.append(f"  {k}: {v:.4f}")
        
        # FIXED: Handle feature importances correctly
        if "feature_importances" in metrics:
            lines.append("Top Feature Importances (Predictive Power):")
            feat_imp = metrics["feature_importances"]
            features_list = st.session_state.model_features_list
            
            # Convert to dictionary if it's a list
            if isinstance(feat_imp, list):
                # Handle case where lengths don't match
                min_len = min(len(features_list), len(feat_imp))
                importances_dict = dict(zip(features_list[:min_len], feat_imp[:min_len]))
            elif isinstance(feat_imp, dict):
                importances_dict = feat_imp
            else:
                importances_dict = {}
            
            # Sort and display top 5
            if importances_dict:
                sorted_imp = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
                for feat, val in sorted_imp[:5]:
                    lines.append(f"  {feat}: {val:.4f}")

    return "\n".join(lines)


def format_ai_response(text: str) -> str:
    """Format AI response with better HTML styling."""
    if not text:
        return "<p>No insights generated yet. Click 'Generate Insights' to begin.</p>"
    
    # Preserve code blocks if any
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"
    
    text = re.sub(r'```.*?```', save_code_block, text, flags=re.DOTALL)
    
    # Format headers
    text = re.sub(r'### (.*?)\n', r'<h3 style="color: #f093fb; margin-top: 1.2rem; margin-bottom: 0.5rem; font-size: 1.2rem;">\1</h3>\n', text)
    text = re.sub(r'## (.*?)\n', r'<h2 style="color: #a78bfa; margin-top: 1.5rem; margin-bottom: 0.75rem; font-size: 1.4rem; border-left: 3px solid #a78bfa; padding-left: 0.8rem;">\1</h2>\n', text)
    text = re.sub(r'# (.*?)\n', r'<h1 style="color: white; margin-top: 1.5rem; margin-bottom: 0.75rem; font-size: 1.6rem;">\1</h1>\n', text)
    
    # Bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #f093fb;">\1</strong>', text)
    
    # Italic text
    text = re.sub(r'\*(.*?)\*', r'<em style="color: rgba(255,255,255,0.8);">\1</em>', text)
    
    # Handle tables
    lines = text.split('\n')
    formatted_lines = []
    in_table = False
    table_header_detected = False
    
    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_html = '<div style="overflow-x: auto; margin: 1.2rem 0;"><table style="width: 100%; border-collapse: collapse; background: rgba(0,0,0,0.2); border-radius: 12px; overflow: hidden;">'
            
            if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                continue
            
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            
            is_header = False
            if not table_header_detected and any(cell.startswith('**') or cell.endswith('**') or cell.upper() == cell for cell in cells):
                is_header = True
                table_header_detected = True
            
            if is_header:
                table_html += '<thead style="background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));">'
                table_html += '<tr>'
                for cell in cells:
                    cell_clean = re.sub(r'\*\*(.*?)\*\*', r'\1', cell)
                    table_html += f'<th style="border: 1px solid rgba(255,255,255,0.2); padding: 0.75rem; text-align: left; font-weight: 700; color: #f093fb;">{cell_clean}</th>'
                table_html += '</table>'
                table_html += '</thead><tbody>'
            else:
                table_html += '<tr>'
                for cell in cells:
                    cell_clean = re.sub(r'[📊📈📉🔍✅❌⚠️💡🚀👥⚙️📋]', '', cell)
                    cell_clean = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell_clean)
                    table_html += f'<td style="border: 1px solid rgba(255,255,255,0.15); padding: 0.6rem; color: rgba(255,255,255,0.85);">{cell_clean}</td>'
                table_html += '</tr>'
        else:
            if in_table:
                table_html += '</tbody></table></div>'
                formatted_lines.append(table_html)
                in_table = False
                table_header_detected = False
                table_html = ""
            formatted_lines.append(line)
    
    if in_table:
        table_html += '</tbody></table></div>'
        formatted_lines.append(table_html)
    
    text = '\n'.join(formatted_lines)
    
    # Format bullet points
    def format_bullet_point(match):
        content = match.group(1)
        if match.group(0).strip().startswith(tuple('123456789')):
            return f'<li style="margin-bottom: 0.4rem; color: rgba(255,255,255,0.9);">{content}</li>'
        else:
            return f'<li style="margin-bottom: 0.4rem; color: rgba(255,255,255,0.9);">• {content}</li>'
    
    text = re.sub(r'^[\-\*•]\s+(.*?)$', format_bullet_point, text, flags=re.MULTILINE)
    text = re.sub(r'^[0-9]+\.\s+(.*?)$', format_bullet_point, text, flags=re.MULTILINE)
    
    text = re.sub(r'(<li.*?>.*?</li>\n?)+', r'<ul style="margin: 0.8rem 0; padding-left: 1.5rem; list-style-type: none;">\g<0></ul>', text, flags=re.DOTALL)
    
    text = re.sub(r'^---$', r'<hr style="margin: 1.5rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);">', text, flags=re.MULTILINE)
    
    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)
    
    text = text.replace('\n\n', '<br>')
    text = re.sub(r'<br>\s*<h', '<h', text)
    text = re.sub(r'</h\d>\s*<br>', '</h2>', text)
    
    return text


def get_modeling_insights_context() -> str:
    """Generate additional context about modeling results for better insights."""
    if not st.session_state.get("modeling_done"):
        return ""
    
    context = []
    context.append("\n=== MODELING RESULTS DETAILS ===\n")
    
    metrics = st.session_state.model_metrics
    best_model = st.session_state.trained_model_name
    task_type = st.session_state.model_task_type
    target_col = st.session_state.model_target_col
    
    context.append(f"Best performing model: {best_model}")
    context.append(f"Task: {task_type} - predicting '{target_col}'")
    
    # Add best model metrics
    best_metrics = metrics.get(best_model, {})
    if task_type == "classification":
        accuracy = best_metrics.get("Accuracy", 0)
        f1 = best_metrics.get("F1-Score", 0)
        context.append(f"Best model accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        context.append(f"Best model F1-Score: {f1:.4f}")
        
        # Performance interpretation
        if accuracy >= 0.9:
            context.append("Performance: EXCELLENT - Model is highly accurate")
        elif accuracy >= 0.8:
            context.append("Performance: GOOD - Model performs well")
        elif accuracy >= 0.7:
            context.append("Performance: MODERATE - Model has reasonable accuracy")
        else:
            context.append("Performance: NEEDS IMPROVEMENT - Consider feature engineering or more data")
    else:
        r2 = best_metrics.get("R2 Score", 0)
        rmse = best_metrics.get("RMSE", 0)
        context.append(f"Best model R² score: {r2:.4f}")
        context.append(f"Best model RMSE: {rmse:.4f}")
        
        if r2 >= 0.9:
            context.append("Performance: EXCELLENT - Model explains 90%+ variance")
        elif r2 >= 0.7:
            context.append("Performance: GOOD - Model explains 70%+ variance")
        elif r2 >= 0.5:
            context.append("Performance: MODERATE - Model explains 50%+ variance")
        else:
            context.append("Performance: NEEDS IMPROVEMENT - Low explanatory power")
    
    # Feature importance if available
    if "feature_importances" in best_metrics:
        feat_imp = best_metrics["feature_importances"]
        features_list = st.session_state.model_features_list
        
        if isinstance(feat_imp, list):
            min_len = min(len(features_list), len(feat_imp))
            importances_dict = dict(zip(features_list[:min_len], feat_imp[:min_len]))
        else:
            importances_dict = feat_imp
        
        if importances_dict:
            top_features = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            context.append(f"Top 3 influential features: {', '.join([f[0] for f in top_features])}")
    
    return "\n".join(context)


def get_data_quality_insights_context() -> str:
    """Generate context about data quality for better insights."""
    context = []
    
    quality_report = st.session_state.get("quality_report")
    cleaning_report = st.session_state.get("cleaning_report")
    
    if quality_report:
        context.append("\n=== DATA QUALITY CONTEXT ===\n")
        score = quality_report.get('score', 0)
        completeness = quality_report.get('completeness', 100)
        
        if score >= 85:
            context.append("Data Quality: EXCELLENT - High quality dataset ready for analysis")
        elif score >= 70:
            context.append("Data Quality: GOOD - Minor issues that were addressed")
        elif score >= 50:
            context.append("Data Quality: FAIR - Some data quality concerns to note")
        else:
            context.append("Data Quality: POOR - Significant data quality issues")
        
        context.append(f"Completeness: {completeness}% complete")
        
        if quality_report.get('duplicate_rate', 0) > 5:
            context.append("Note: High duplicate rate detected - may affect analysis")
        
        if quality_report.get('missing_cells', 0) > 100:
            context.append("Note: Significant missing values were present and handled")
    
    if cleaning_report:
        before_rows = cleaning_report.get('before_shape', [0, 0])[0]
        after_rows = cleaning_report.get('after_shape', [0, 0])[0]
        rows_removed = before_rows - after_rows
        
        if rows_removed > 0:
            context.append(f"Cleaning removed {rows_removed} rows ({rows_removed/before_rows*100:.1f}% of data)")
        
        if cleaning_report.get('log'):
            context.append(f"Applied {len(cleaning_report['log'])} cleaning operations")
    
    return "\n".join(context)


def render():
    """Main render function for AI Insights page."""
    
    st.markdown("## 💡 AI Insights Agent")
    st.markdown("Generates intelligent, business-level insights by analyzing your data quality, statistics, and machine learning models.")
    st.markdown("---")

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
    <div style="background:linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
                border:1px solid rgba(102,126,234,0.5);
                border-radius:12px;
                padding:0.7rem 1rem;
                margin-bottom:1.5rem;">
        <span style="color:rgba(255,255,255,0.8);font-size:0.85rem;">🤖 Active Model: </span>
        <code style="color:#a78bfa;font-size:0.85rem;background:rgba(0,0,0,0.3);padding:2px 8px;border-radius:6px;">
            {model_name}
        </code>
        <span style="color:#34d399;font-size:0.75rem;margin-left:0.8rem;">✓ OpenRouter</span>
    </div>
    """, unsafe_allow_html=True)

    # Show what data is available for insights
    st.markdown("### 📊 Available Data for Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        has_quality = st.session_state.get("quality_report") is not None
        st.markdown(f"{'✅' if has_quality else '⏳'} Quality Report: {'Available' if has_quality else 'Not yet'}")
    with col2:
        has_cleaning = st.session_state.get("cleaning_report") is not None
        st.markdown(f"{'✅' if has_cleaning else '⏳'} Cleaning Report: {'Available' if has_cleaning else 'Not yet'}")
    with col3:
        has_modeling = st.session_state.get("modeling_done", False)
        st.markdown(f"{'✅' if has_modeling else '⏳'} Modeling Results: {'Available' if has_modeling else 'Not yet'}")
    
    st.markdown("---")

    # Insight type selector
    insight_options = {
        "General business insights": "📊",
        "Anomalies and red flags": "⚠️",
        "Operational efficiency observations": "⚙️",
        "Executive summary (non-technical)": "📋",
        "Model performance analysis": "🤖",
        "Data quality recommendations": "🔍"
    }
    # Adjust options based on available data
    if not has_modeling:
        insight_options.pop("Model performance analysis", None)
    
    insight_type = st.selectbox(
        "Select insight type",
        list(insight_options.keys()),
        format_func=lambda x: f"{insight_options[x]} {x}"
    )

    extra_context = st.text_area(
        "Additional context (optional)",
        placeholder="Example: This is monthly sales data from an e-commerce company targeting Southeast Asia. We're interested in predicting customer churn...",
        height=80,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Generate Insights", use_container_width=True, type="primary")

    if run_btn:
        # Build comprehensive context
        data_summary = build_data_summary(df)
        quality_context = get_data_quality_insights_context()
        modeling_context = get_modeling_insights_context()
        
        # Combine all context
        full_context = data_summary + quality_context + modeling_context
        
        # Customize system prompt based on insight type
        system_prompt = f"""You are a senior data analyst and business intelligence expert specializing in extracting actionable insights from data.

Current focus: {insight_type}

Guidelines:
1. Be specific - reference actual column names, values, and metrics from the data
2. Be quantitative - use actual numbers from the analysis
3. Be actionable - provide clear recommendations
4. Be professional - use clear business language
5. Use markdown formatting with ## for sections, ### for subsections
6. Use **bold** for key metrics and important numbers

Available analysis includes:
- Dataset statistics and correlations
- Data quality assessment and cleaning results  
- {'Machine learning model performance and feature importance' if has_modeling else ''}

Write insights that are directly grounded in the provided data. Avoid generic statements."""

        user_prompt = f"""Analyze the following dataset and provide {insight_type.lower()}.

{extra_context if extra_context.strip() else ''}

=== DATASET AND ANALYSIS SUMMARY ===
{full_context}

Please provide a comprehensive response structured as:

## Key Findings
- 3-5 specific findings with actual numbers and **bold** for important metrics

## Patterns & Trends
- What stands out in the data
- Relationships between variables
- {'Model performance patterns and what they reveal' if has_modeling else ''}

## Actionable Recommendations
- 2-4 concrete recommendations based on the data
- Prioritized by potential impact

## {'Model Improvement Suggestions' if has_modeling else 'Data Quality Notes'}
- {'Specific ways to improve model performance' if has_modeling else 'Any data quality concerns worth addressing'}

Keep the total response under 700 words. Be specific, data-driven, and actionable.
Use markdown formatting. Do not use emojis inside table cells if you create tables."""

        with st.spinner(f"🧠 Analyzing with {model_name}..."):
            response = call_llm(user_prompt, system_prompt, max_tokens=1800)
            st.session_state.insights_text = response
            st.session_state.insights_type = insight_type
            st.rerun()

    # Display insights
    if st.session_state.get("insights_text"):
        st.markdown("---")
        
        insight_icon = insight_options.get(st.session_state.get('insights_type', ''), "💡")
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span style="font-size: 2.5rem;">{insight_icon}</span>
            <h2 style="display: inline-block; margin-left: 0.5rem; margin-bottom: 0;">{st.session_state.get('insights_type', 'AI Insights')}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        formatted_response = format_ai_response(st.session_state.insights_text)
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(102,126,234,0.3);
                    border-radius: 20px;
                    padding: 2rem;
                    line-height: 1.7;
                    font-size: 0.95rem;">
            {formatted_response}
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                "📥 Download Insights",
                data=st.session_state.insights_text,
                file_name=f"insights_{st.session_state.get('insights_type', '').replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state.insights_text = None
                st.rerun()
        
        with col3:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.info("Select and copy the text above (Ctrl+C / Cmd+C)")
        
        with col4:
            if st.button("➡️ Go to Report", use_container_width=True):
                st.session_state.current_page = "report"
                st.rerun()

        # Follow-up Question Section
        st.markdown("---")
        st.markdown("### ❓ Ask a Follow-up Question")
        st.markdown("Get deeper insights about specific aspects of your data or models.")
        
        # Pre-defined follow-up questions based on available data
        quick_questions = []
        
        if st.session_state.get("modeling_done"):
            quick_questions.extend([
                "What features most influence predictions?",
                "How can we improve model accuracy?",
                "Is the model ready for production?"
            ])
        
        if st.session_state.get("quality_report"):
            quick_questions.extend([
                "What data quality issues should we fix?",
                "How reliable are these insights?"
            ])
        
        if quick_questions:
            st.markdown("**Quick questions:**")
            cols = st.columns(len(quick_questions))
            for idx, q in enumerate(quick_questions):
                with cols[idx]:
                    if st.button(q, key=f"quick_q_{idx}", use_container_width=True):
                        st.session_state.followup_question = q
                        st.rerun()
        
        # Custom question input
        col_q1, col_q2 = st.columns([4, 1])
        with col_q1:
            default_q = st.session_state.get("followup_question", "")
            custom_q = st.text_input(
                "",
                value=default_q,
                placeholder="Example: What are the top 3 risks based on this data?",
                label_visibility="collapsed",
                key="custom_question"
            )
        with col_q2:
            if st.button("Ask", key="custom_ask", use_container_width=True):
                if custom_q.strip():
                    # Clear the stored question after using
                    if "followup_question" in st.session_state:
                        del st.session_state.followup_question
                    
                    # Build context for follow-up
                    data_summary = build_data_summary(df)
                    quality_context = get_data_quality_insights_context()
                    modeling_context = get_modeling_insights_context()
                    
                    full_context = data_summary + quality_context + modeling_context
                    
                    prompt = f"""Dataset analysis summary:
{full_context}

Follow-up Question: {custom_q}

Provide a clear, specific, data-driven answer. Use markdown formatting.
Keep the answer focused and under 400 words.
Use **bold** for important numbers and findings.
Be actionable and specific - reference actual columns and metrics from the data."""

                    with st.spinner("💭 Generating answer..."):
                        answer = call_llm(prompt, max_tokens=1000)
                        
                        formatted_answer = answer.replace('**', '<strong>').replace('**', '</strong>')
                        formatted_answer = formatted_answer.replace('###', '<h4 style="color: #f093fb; margin: 0.5rem 0 0.25rem 0;">')
                        formatted_answer = formatted_answer.replace('\n\n', '<br><br>')
                        formatted_answer = formatted_answer.replace('- ', '• ')
                        
                        # Convert numbered lists
                        formatted_answer = re.sub(r'(\d+)\.\s+', r'<strong>\1.</strong> ', formatted_answer)
                        
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg, rgba(100,108,255,0.12), rgba(167,139,250,0.06));
                                    border-left: 4px solid #6468ff;
                                    border-radius: 12px;
                                    padding: 1.2rem;
                                    margin-top: 0.75rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                                <span style="font-size: 1.2rem;">💬</span>
                                <span style="color: #a78bfa; font-weight: 700;">Answer:</span>
                            </div>
                            <div style="color: rgba(255,255,255,0.9); line-height: 1.7;">
                                {formatted_answer}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter a question.")

        # Disclaimer
        st.markdown("---")
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.1);
                    border-radius:12px;
                    padding:0.8rem;
                    text-align: center;">
            <p style="color:rgba(255,255,255,0.5); margin:0; font-size:0.75rem;">
                🤖 AI-generated insights using OpenRouter API | Based on your data quality assessment, cleaning results, and modeling performance
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("← Back to Modeling", use_container_width=True):
                st.session_state.current_page = "modeling"
                st.rerun()
        with col_nav2:
            if st.button("➡️ Generate Full Report", use_container_width=True):
                st.session_state.current_page = "report"
                st.rerun()

    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; background: linear-gradient(135deg, #fff, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Ready to Generate AI Insights
            </div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6); max-width: 500px; margin: 0 auto;">
                Select an insight type and click "Generate Insights" to get AI-powered analysis based on:
            </div>
            <div style="margin-top: 1rem; text-align: left; max-width: 400px; margin-left: auto; margin-right: auto;">
                <li>✓ Your dataset structure and statistics</li>
                <li>✓ Data quality assessment results</li>
                <li>✓ Cleaning operations performed</li>
                <li>✓ Model performance metrics (if modeling done)</li>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
