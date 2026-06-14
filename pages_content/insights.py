import streamlit as st
import pandas as pd
import numpy as np
from utils.session import get_df, is_data_loaded
from utils.llm import call_llm, get_active_model
import re


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
        lines.append("=== Machine Learning Modeling Summary ===")
        lines.append(f"Target Column: {st.session_state.model_target_col}")
        lines.append(f"Features Used: {', '.join(st.session_state.model_features_list)}")
        lines.append(f"Task Type: {st.session_state.model_task_type}")
        lines.append(f"Best Trained Model: {st.session_state.trained_model_name}")
        lines.append("Model Performance Metrics on Test Set:")
        metrics = st.session_state.model_metrics[st.session_state.trained_model_name]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                lines.append(f"  {k}: {v:.4f}")
        
        # Feature importances
        if "feature_importances" in metrics:
            lines.append("Top Feature Importances (Predictive Power):")
            feat_imp = metrics["feature_importances"]
            sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
            for feat, val in sorted_imp[:5]:
                lines.append(f"  {feat}: {val:.4f}")

    return "\n".join(lines)


def format_ai_response(text: str) -> str:
    """Format AI response with better HTML styling."""
    
    # First, preserve code blocks if any
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
                table_html += '</tr>'
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


def render():
    st.markdown("## 💡 AI Insights Agent")
    st.markdown("Uses an LLM via OpenRouter to generate business-level insights from your data.")
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

        system_prompt = """You are a senior data analyst and business intelligence expert. 
You analyze datasets and provide clear, actionable, business-focused insights.
Write in a professional but accessible tone. Use bullet points and short paragraphs.
Always be specific and reference actual column names and values from the data.
Format your response with clear sections using markdown (## for sections, ### for subsections, - for bullet points).
Use **bold** for key metrics and important numbers.
Avoid generic statements. Every insight must be grounded in the data provided.

For tables, use standard markdown format:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| value 1 | value 2 | value 3 |

Do not add emoji icons inside table cells."""

        user_prompt = f"""Analyze the following dataset and provide {insight_type.lower()}.

{extra_context if extra_context.strip() else ''}

=== DATASET SUMMARY ===
{data_summary}

Please provide a well-structured response with:

## Key Findings
- 3-5 bullet points with specific numbers and **bold** for important metrics

## Patterns & Trends
- What stands out in the data
- Use a table if comparing multiple categories (without emojis in cells)

## Actionable Recommendations
- 2-3 concrete next steps with clear justifications

## Data Quality Notes
- Any concerns worth flagging

Keep the total response under 600 words. Be specific and data-driven.
Use markdown formatting for better readability but avoid emojis inside table cells."""

        with st.spinner(f"🧠 Analyzing with {model_name}..."):
            response = call_llm(user_prompt, system_prompt, max_tokens=1500)
            st.session_state.insights_text = response
            st.session_state.insights_type = insight_type

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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                "📥 Download",
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
            if st.button("📋 Copy", use_container_width=True):
                st.info("Select and copy the text above (Ctrl+C / Cmd+C)")
        
        with col4:
            if st.button("➡️ Report", use_container_width=True):
                st.session_state.current_page = "report"
                st.rerun()

        # IMPROVED Follow-up Question Section
        st.markdown("---")
        st.markdown("### ❓ Ask a Follow-up Question")
        
        col_q1, col_q2 = st.columns([4, 1])
        with col_q1:
            custom_q = st.text_input(
                "",
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

Answer the question in a clear, well-structured format. Use markdown for better readability:
- Use **bold** for important numbers and metrics
- Use bullet points for lists
- Use numbered steps for action items
- Keep the answer concise but informative (max 300 words)"""

                    with st.spinner("💭 Generating answer..."):
                        answer = call_llm(prompt, max_tokens=800)
                        
                        # Clean and format the answer
                        answer = answer.replace('**', '<strong>').replace('**', '</strong>')
                        answer = answer.replace('###', '<h4 style="color: #f093fb; margin: 0.5rem 0 0.25rem 0;">')
                        answer = answer.replace('\n\n', '<br><br>')
                        answer = answer.replace('- ', '• ')
                        
                        # Convert numbered lists
                        answer = re.sub(r'(\d+)\.\s+', r'<strong>\1.</strong> ', answer)
                        
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg, rgba(100,108,255,0.12), rgba(167,139,250,0.06));
                                    border-left: 4px solid #6468ff;
                                    border-radius: 12px;
                                    padding: 1.2rem;
                                    margin-top: 0.75rem;
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                                <span style="font-size: 1.2rem;">💬</span>
                                <span style="color: #a78bfa; font-weight: 700; font-size: 0.9rem;">Answer:</span>
                            </div>
                            <div style="color: rgba(255,255,255,0.9); line-height: 1.7; font-size: 0.9rem;">
                                {answer}
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
                🤖 AI-generated insights | Always verify critical findings with domain experts
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("← Back to Visualization", use_container_width=True):
                st.session_state.current_page = "visualization"
                st.rerun()
        with col_nav2:
            if st.button("➡️ Proceed to Report", use_container_width=True):
                st.session_state.current_page = "report"
                st.rerun()

    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; background: linear-gradient(135deg, #fff, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Ready to Generate Insights
            </div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6);">
                Select an insight type and click "Generate Insights" to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Back to Home"):
            st.session_state.current_page = "home"
            st.rerun()
