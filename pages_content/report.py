import streamlit as st
import pandas as pd
from datetime import datetime
from utils.session import get_df, get_raw_df, is_data_loaded
from utils.llm import call_llm, get_active_model
import io
import re


def build_full_context(df: pd.DataFrame) -> str:
    """Build comprehensive context for the report."""
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

    # Safe access to quality report
    quality_report = st.session_state.get("quality_report")
    if quality_report and isinstance(quality_report, dict):
        score = quality_report.get('score', 'N/A')
        completeness = quality_report.get('completeness', 'N/A')
        duplicate_rows = quality_report.get('duplicate_rows', 'N/A')
        parts.append(f"\nData Quality Score: {score}/100")
        parts.append(f"Completeness: {completeness}%")
        parts.append(f"Duplicates: {duplicate_rows} rows")

    insights = st.session_state.get("insights_text")
    if insights:
        parts.append(f"\nAI Insights Summary:\n{insights[:800]}")

    cleaning_report = st.session_state.get("cleaning_report")
    if cleaning_report and isinstance(cleaning_report, dict):
        before = cleaning_report.get('before_shape', [0, 0])[0] if cleaning_report.get('before_shape') else 0
        after = cleaning_report.get('after_shape', [0, 0])[0] if cleaning_report.get('after_shape') else 0
        parts.append(f"\nCleaning Summary:")
        parts.append(f"  - Rows before: {before}")
        parts.append(f"  - Rows after: {after}")
        parts.append(f"  - Rows removed: {before - after}")

    return "\n".join(parts)


def format_report_text(text: str) -> str:
    """Format report text with better HTML styling."""
    
    # Format main headers
    text = re.sub(r'^# (.*?)$', r'<h1 style="color: #f093fb; font-size: 2rem; margin-top: 0; margin-bottom: 0.5rem; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;">\1</h1>', text, flags=re.MULTILINE)
    
    # Format section headers
    text = re.sub(r'^## (.*?)$', r'<h2 style="color: #a78bfa; font-size: 1.5rem; margin-top: 1.5rem; margin-bottom: 0.75rem; border-left: 4px solid #a78bfa; padding-left: 0.8rem;">\1</h2>', text, flags=re.MULTILINE)
    
    # Format sub-section headers
    text = re.sub(r'^### (.*?)$', r'<h3 style="color: #c4b5fd; font-size: 1.2rem; margin-top: 1rem; margin-bottom: 0.5rem;">\1</h3>', text, flags=re.MULTILINE)
    
    # Bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #f093fb;">\1</strong>', text)
    
    # Italic text
    text = re.sub(r'\*(.*?)\*', r'<em style="color: rgba(255,255,255,0.8);">\1</em>', text)
    
    # Format tables
    lines = text.split('\n')
    formatted_lines = []
    in_table = False
    table_html = ""
    table_captured = []
    
    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_html = '<div style="overflow-x: auto; margin: 1.2rem 0;"><table style="width: 100%; border-collapse: collapse; background: rgba(0,0,0,0.2); border-radius: 12px; overflow: hidden;">'
            
            # Skip separator lines
            if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                continue
            
            # Process table row
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            
            # Determine if header
            if not table_captured:
                table_html += '<thead style="background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));">'
                table_html += '<tr>'
                for cell in cells:
                    cell_clean = re.sub(r'\*\*(.*?)\*\*', r'\1', cell)
                    table_html += f'<th style="border: 1px solid rgba(255,255,255,0.2); padding: 0.75rem; text-align: left; font-weight: 700; color: #f093fb;">{cell_clean}</th>'
                table_html += '</tr>'
                table_html += '</thead><tbody>'
                table_captured.append(True)
            else:
                table_html += '<tr>'
                for cell in cells:
                    cell_clean = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                    table_html += f'<td style="border: 1px solid rgba(255,255,255,0.15); padding: 0.6rem; color: rgba(255,255,255,0.85);">{cell_clean}</td>'
                table_html += '</tr>'
        else:
            if in_table:
                table_html += '</tbody></table></div>'
                formatted_lines.append(table_html)
                in_table = False
                table_captured = []
                table_html = ""
            formatted_lines.append(line)
    
    if in_table:
        table_html += '</tbody></table></div>'
        formatted_lines.append(table_html)
    
    text = '\n'.join(formatted_lines)
    
    # Format bullet points
    text = re.sub(r'^[\-\*]\s+(.*?)$', r'<li style="margin-bottom: 0.5rem; color: rgba(255,255,255,0.9);">• \1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^[0-9]+\.\s+(.*?)$', r'<li style="margin-bottom: 0.5rem; color: rgba(255,255,255,0.9);"><strong style="color: #f093fb;">✓</strong> \1</li>', text, flags=re.MULTILINE)
    
    # Wrap consecutive list items
    text = re.sub(r'(<li.*?>.*?</li>\n?)+', r'<ul style="margin: 0.8rem 0; padding-left: 1.5rem; list-style-type: none;">\g<0></ul>', text, flags=re.DOTALL)
    
    # Format horizontal rules
    text = re.sub(r'^---$', r'<hr style="margin: 1.5rem 0; border: none; height: 2px; background: linear-gradient(90deg, transparent, rgba(102,126,234,0.5), rgba(118,75,162,0.5), transparent);">', text, flags=re.MULTILINE)
    
    # Add spacing for paragraphs
    text = text.replace('\n\n', '<br><br>')
    
    return text


def generate_markdown_report(df: pd.DataFrame, llm_report: str, file_name: str) -> str:
    """Generate complete markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    quality_report = st.session_state.get("quality_report", {})
    insights_text = st.session_state.get("insights_text", "Not generated.")

    lines = []
    lines.append(f"# 📊 Data Analysis Report")
    lines.append(f"")
    lines.append(f"**File:** `{file_name}`  ")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Tool:** CSV Insight Agents  ")
    lines.append(f"**Dataset Size:** {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    lines.append("---\n")

    lines.append("## 📋 1. Dataset Overview\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Rows | **{df.shape[0]:,}** |")
    lines.append(f"| Total Columns | **{df.shape[1]}** |")
    lines.append(f"| Numeric Columns | **{len(numeric_cols)}** |")
    lines.append(f"| Categorical Columns | **{len(cat_cols)}** |")
    lines.append(f"| Missing Cells | **{df.isnull().sum().sum():,}** |")
    lines.append(f"| Duplicate Rows | **{df.duplicated().sum()}** |\n")

    # Column list
    lines.append("### 📝 Column Details\n")
    for i, col in enumerate(df.columns[:10], 1):
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        lines.append(f"- **{col}**: `{dtype}` (Missing: {missing} / {missing_pct:.1f}%)")
    
    if len(df.columns) > 10:
        lines.append(f"- ... and {len(df.columns) - 10} more columns")

    lines.append("\n")

    if quality_report:
        lines.append("## ✅ 2. Data Quality Assessment\n")
        
        score = quality_report.get('score', 0)
        if score >= 85:
            grade = "🟢 Excellent"
        elif score >= 70:
            grade = "🟡 Good"
        elif score >= 50:
            grade = "🟠 Fair"
        else:
            grade = "🔴 Poor"
        
        lines.append(f"### Overall Quality Score: **{score}/100** ({grade})\n")
        lines.append("| Dimension | Score | Status |")
        lines.append("|-----------|-------|--------|")
        lines.append(f"| Completeness | {quality_report.get('completeness', 0)}% | {'✅' if quality_report.get('completeness', 0) > 90 else '⚠️'} |")
        lines.append(f"| Duplicate Rate | {quality_report.get('duplicate_rate', 0)}% | {'✅' if quality_report.get('duplicate_rate', 0) < 5 else '⚠️'} |\n")

    if numeric_cols:
        lines.append("\n## 📊 3. Statistical Summary\n")
        desc = df[numeric_cols].describe().round(2)
        lines.append("```")
        lines.append(desc.to_string())
        lines.append("```")
        lines.append("\n")

    lines.append("## 💡 4. AI-Generated Insights\n")
    lines.append(insights_text)
    lines.append("\n")

    lines.append("## 📄 5. Detailed Analysis\n")
    lines.append(llm_report)
    lines.append("\n")

    lines.append("---")
    lines.append(f"*Report generated by CSV Insight Agents · {now}*")
    lines.append("*Powered by Streamlit, Plotly, and OpenRouter AI*")

    return "\n".join(lines)


def export_to_csv(dataframe):
    """Export dataframe to CSV format."""
    return dataframe.to_csv(index=False)


def export_to_excel(dataframe):
    """Export dataframe to Excel format."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, sheet_name="Report_Data", index=False)
    return output.getvalue()


def export_to_json(dataframe):
    """Export dataframe to JSON format."""
    return dataframe.to_json(orient="records", indent=2)


def export_to_html(dataframe, title="Data Export"):
    """Export dataframe to HTML format."""
    return dataframe.to_html(index=False, border=0, classes='dataframe')


def render():
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
    st.markdown("Compiles all agent outputs into a comprehensive, downloadable analysis report.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("⚠️ No data loaded. Please upload a CSV on the Home page first.")
        if st.button("← Back to Home", use_container_width=False):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()
    file_name = st.session_state.get("file_name", "data.csv")

    # Pipeline status
    st.markdown("### 📋 Pipeline Completion Status")
    
    steps = [
        ("🔍 Data Quality", st.session_state.get("quality_report") is not None),
        ("🧹 Data Cleaning", st.session_state.get("cleaning_report") is not None),
        ("📊 Statistical Analysis", st.session_state.get("stats_done", False)),
        ("📈 Visualization", st.session_state.get("viz_done", False)),
        ("💡 AI Insights", st.session_state.get("insights_text") is not None),
    ]

    cols = st.columns(len(steps))
    for col, (label, done) in zip(cols, steps):
        color = "#34d399" if done else "#475569"
        bg_color = "rgba(52,211,153,0.1)" if done else "rgba(71,85,105,0.1)"
        icon = "✅" if done else "⏳"
        col.markdown(f"""
        <div style="text-align:center;padding:0.75rem;background:{bg_color};
                    border-radius:12px;border:1px solid {color};">
            <div style="font-size:1.5rem;">{icon}</div>
            <div style="font-size:0.75rem;color:{color};margin-top:0.3rem;font-weight:600;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    completed = sum(1 for _, done in steps if done)
    progress = completed / len(steps) if len(steps) > 0 else 0
    
    st.progress(progress)
    st.caption(f"📌 {completed}/{len(steps)} steps completed - {int(progress*100)}% ready for report generation")

    st.markdown("---")

    # Report options
    st.markdown("### ⚙️ Report Configuration")
    
    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        report_tone = st.selectbox(
            "🎨 Report Tone",
            ["Executive (non-technical)", "Technical (data science)", "Mixed (balanced)"],
            help="Choose the writing style that best suits your audience"
        )
    with col_opts2:
        include_recs = st.checkbox("💡 Include Recommendations Section", value=True, help="Add actionable recommendations to the report")

    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_btn = st.button("🚀 Generate Full Report", use_container_width=True, type="primary")

    if generate_btn:
        try:
            context = build_full_context(df)

            system_prompt = f"""You are a senior data analyst writing a professional analysis report.
Tone: {report_tone}.
Write clearly, use structure (headings, bullet points), and be specific with numbers.
This report will be read by {'business stakeholders' if 'Executive' in report_tone else 'technical teams'}.
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

                full_report = generate_markdown_report(df, llm_report, file_name)
                st.session_state.full_report_md = full_report

            st.success("✅ Report generated successfully!")
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.info("Please make sure you have completed the previous agents or try again.")

    # Display report
    if st.session_state.get("report_text"):
        st.markdown("---")
        st.markdown("### 📄 Generated Report")
        
        # Format and display the report
        formatted_report = format_report_text(st.session_state.report_text)
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(102,126,234,0.3);
                    border-radius: 20px;
                    padding: 2rem;
                    max-height: 600px;
                    overflow-y: auto;">
            {formatted_report}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Export Options - Multi-format
        st.markdown("### 📤 Export Options")
        st.markdown("Download the cleaned data in your preferred format:")
        
        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

        with col_dl1:
            csv_data = export_to_csv(df)
            st.download_button(
                "📄 CSV",
                data=csv_data,
                file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_dl2:
            excel_data = export_to_excel(df)
            st.download_button(
                "📊 Excel",
                data=excel_data,
                file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col_dl3:
            json_data = export_to_json(df)
            st.download_button(
                "📋 JSON",
                data=json_data,
                file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_dl4:
            html_data = export_to_html(df, title="Data Analysis Report")
            st.download_button(
                "🌐 HTML",
                data=html_data,
                file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True,
            )

        # Also download the report itself
        st.markdown("### 📝 Report Downloads")
        col_rep1, col_rep2 = st.columns(2)
        
        with col_rep1:
            st.download_button(
                "📄 Download Report (Markdown)",
                data=st.session_state.get("full_report_md", ""),
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        
        with col_rep2:
            st.download_button(
                "📝 Download Report (Text)",
                data=st.session_state.get("report_text", ""),
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # Completion celebration
        st.markdown("---")
        st.markdown("""
        <div class="status-complete">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 2rem;">🎉</span>
                    <span style="font-weight: 700; font-size: 1.2rem; margin-left: 0.5rem;">Analysis Complete!</span>
                </div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                    All agents have been executed successfully
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        with col_nav1:
            if st.button("🏠 Start New Analysis", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col_nav2:
            if st.button("🔄 Regenerate Report", use_container_width=True):
                st.session_state.report_text = None
                st.rerun()
        with col_nav3:
            if st.button("💡 Back to Insights", use_container_width=True):
                st.session_state.current_page = "insights"
                st.rerun()
            
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; background: linear-gradient(135deg, #fff, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Ready to Generate Report
            </div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6); max-width: 400px; margin: 0 auto;">
                Complete the previous agents (Quality, Cleaning, Stats, Visualization, Insights) for the best results, or click Generate to create a report with available data.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
