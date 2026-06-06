import streamlit as st
import pandas as pd
import numpy as np
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

    return "\n".join(lines)


def render():
    st.markdown("## 💡 AI Insights Agent")
    st.markdown("Uses an LLM via OpenRouter to generate business-level insights from your data.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()

    # Model info
    model_name = get_active_model()
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.1);
                border:1px solid rgba(255,255,255,0.2);
                border-radius:12px;
                padding:0.7rem 1rem;
                margin-bottom:1rem;
                display:inline-block;">
        <span style="color:rgba(255,255,255,0.8);font-size:0.8rem;">🤖 Model: </span>
        <code style="color:#a78bfa;font-size:0.8rem;">{model_name}</code>
    </div>
    """, unsafe_allow_html=True)

    # Insight type selector
    insight_type = st.selectbox(
        "What kind of insights do you want?",
        [
            "General business insights",
            "Anomalies and red flags",
            "Growth opportunities",
            "Customer behavior patterns",
            "Operational efficiency observations",
            "Executive summary (non-technical)",
        ]
    )

    extra_context = st.text_area(
        "Additional context (optional)",
        placeholder="e.g. This is monthly sales data from an e-commerce company targeting Southeast Asia...",
        height=80,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("Generate AI Insights", use_container_width=True)

    if run_btn:
        data_summary = build_data_summary(df)

        system_prompt = """You are a senior data analyst and business intelligence expert. 
You analyze datasets and provide clear, actionable, business-focused insights.
Write in a professional but accessible tone. Use bullet points and short paragraphs.
Always be specific and reference actual column names and values from the data.
Avoid generic statements. Every insight must be grounded in the data provided."""

        user_prompt = f"""Analyze the following dataset and provide {insight_type.lower()}.

{extra_context if extra_context.strip() else ''}

=== DATASET SUMMARY ===
{data_summary}

Please provide:
1. Key Findings (3-5 bullet points with specific numbers)
2. Patterns and Trends (what stands out in the data)
3. Actionable Recommendations (2-3 concrete next steps)
4. Data Quality Notes (any concerns worth flagging)

Keep the total response under 500 words. Be specific and data-driven."""

        with st.spinner(f"Analyzing with {model_name}..."):
            response = call_llm(user_prompt, system_prompt, max_tokens=1200)
            st.session_state.insights_text = response
            st.session_state.insights_type = insight_type

    # Display insights
    if st.session_state.get("insights_text"):
        st.markdown("---")
        st.markdown(f"#### AI Insights - {st.session_state.get('insights_type', '')}")

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);
                    border:1px solid rgba(255,255,255,0.2);
                    border-radius:12px;
                    padding:1.5rem;
                    line-height:1.7;">
            {st.session_state.insights_text.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

        # Download insights
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "Download Insights as .txt",
            data=st.session_state.insights_text,
            file_name="ai_insights.txt",
            mime="text/plain",
        )

        # Custom question
        st.markdown("---")
        st.markdown("#### Ask a Custom Question")
        custom_q = st.text_input("Ask anything about your data...", placeholder="e.g. Which product category has the highest average rating?")

        if st.button("Ask", key="custom_ask"):
            if custom_q.strip():
                data_summary = build_data_summary(df)
                prompt = f"""Dataset summary:
{data_summary}

Question: {custom_q}

Answer concisely and specifically using the data above."""
                with st.spinner("Thinking..."):
                    answer = call_llm(prompt, max_tokens=600)
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.05);
                                border-left:3px solid #34d399;
                                border-radius:6px;
                                padding:1rem;
                                margin-top:0.5rem;">
                        {answer.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")

    else:
        st.info("Select an insight type and click Generate AI Insights.")

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.2);
                border-radius:10px;
                padding:1rem;">
        <p style="color:rgba(255,255,255,0.7);margin:0;">
            Next step: head to Report to compile and download everything.
        </p>
    </div>
    """, unsafe_allow_html=True)
