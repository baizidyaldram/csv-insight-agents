import streamlit as st
import pandas as pd
import numpy as np
from utils.session import get_df, is_data_loaded


def compute_quality_report(df: pd.DataFrame) -> dict:
    total_cells = df.size
    total_rows = len(df)
    total_cols = len(df.columns)

    # Completeness: % of non-missing cells
    missing_cells = df.isnull().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100

    # Duplicate rate
    duplicate_rows = df.duplicated().sum()
    duplicate_rate = (duplicate_rows / total_rows) * 100

    # Validity: rows where numeric cols are within 3 std devs (no extreme outliers)
    numeric_cols = df.select_dtypes(include="number").columns
    outlier_flags = pd.Series([False] * total_rows, index=df.index)
    outlier_details = {}
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            col_outliers = ((df[col] - mean).abs() > 3 * std)
            outlier_count = col_outliers.sum()
            if outlier_count > 0:
                outlier_details[col] = int(outlier_count)
            outlier_flags = outlier_flags | col_outliers
    validity = (1 - outlier_flags.sum() / total_rows) * 100

    # Consistency: penalise columns with mixed types or suspicious patterns
    inconsistency_count = 0
    inconsistency_details = {}
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        # Check if column looks numeric but stored as string
        numeric_like = pd.to_numeric(non_null, errors="coerce").notna().sum()
        ratio = numeric_like / len(non_null)
        if 0.1 < ratio < 0.9:
            inconsistency_count += 1
            inconsistency_details[col] = f"{ratio*100:.0f}% numeric-like values in text column"
    consistency = max(0, (1 - inconsistency_count / max(total_cols, 1)) * 100)

    # Overall score: weighted average
    score = (
        completeness * 0.35 +
        (100 - duplicate_rate) * 0.20 +
        validity * 0.25 +
        consistency * 0.20
    )

    # Per-column missing summary
    missing_per_col = df.isnull().sum()
    missing_per_col = missing_per_col[missing_per_col > 0].sort_values(ascending=False)

    return {
        "score": round(score, 1),
        "completeness": round(completeness, 1),
        "duplicate_rate": round(duplicate_rate, 1),
        "duplicate_rows": int(duplicate_rows),
        "validity": round(validity, 1),
        "consistency": round(consistency, 1),
        "missing_cells": int(missing_cells),
        "total_cells": int(total_cells),
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_per_col": missing_per_col,
        "outlier_details": outlier_details,
        "inconsistency_details": inconsistency_details,
    }


def score_color(val: float) -> str:
    if val >= 85:
        return "#34d399"   # green
    elif val >= 65:
        return "#fbbf24"   # yellow
    else:
        return "#f87171"   # red


def score_label(val: float) -> str:
    if val >= 85:
        return "✅ Good"
    elif val >= 65:
        return "⚠️ Fair"
    else:
        return "❌ Poor"


def render():
    st.markdown("## 🔍 Data Quality Agent")
    st.markdown("Automated quality assessment — no AI required, runs entirely on your data.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()

    if st.button("▶ Run Data Quality Analysis", use_container_width=False):
        with st.spinner("Analyzing data quality..."):
            report = compute_quality_report(df)
            st.session_state.quality_report = report

    if st.session_state.get("quality_report") is None:
        st.info("Click **Run Data Quality Analysis** to generate the report.")
        return

    r = st.session_state.quality_report

    # ── Overall score ─────────────────────────────────────────────────────────
    score_col = score_color(r["score"])
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e1e2e,#16213e);border:1px solid {score_col}44;
                border-radius:16px;padding:2rem;text-align:center;margin-bottom:1.5rem;">
        <div style="font-size:0.85rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">
            Overall Data Quality Score
        </div>
        <div style="font-size:4rem;font-weight:800;color:{score_col};line-height:1;">
            {r['score']}<span style="font-size:2rem;">/100</span>
        </div>
        <div style="margin-top:0.5rem;font-size:1rem;color:{score_col};">
            {score_label(r['score'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Four metric cards ─────────────────────────────────────────────────────
    cols = st.columns(4)
    metrics = [
        ("Completeness", r["completeness"], "%"),
        ("Validity", r["validity"], "%"),
        ("Consistency", r["consistency"], "%"),
        ("Duplicate Rate", r["duplicate_rate"], "%"),
    ]
    for col, (label, val, unit) in zip(cols, metrics):
        color = score_color(val if label != "Duplicate Rate" else 100 - val)
        col.markdown(f"""
        <div class="metric-box">
            <div class="value" style="color:{color};">{val}{unit}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detail breakdown ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📋 Summary")
        summary_data = {
            "Metric": ["Total Rows", "Total Columns", "Missing Cells", "Duplicate Rows", "Outlier Columns"],
            "Value": [
                f"{r['total_rows']:,}",
                f"{r['total_cols']}",
                f"{r['missing_cells']:,} ({100 - r['completeness']:.1f}%)",
                f"{r['duplicate_rows']:,} ({r['duplicate_rate']:.1f}%)",
                f"{len(r['outlier_details'])}",
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### ❗ Missing Values by Column")
        if len(r["missing_per_col"]) > 0:
            missing_df = pd.DataFrame({
                "Column": r["missing_per_col"].index,
                "Missing Count": r["missing_per_col"].values,
                "Missing %": (r["missing_per_col"].values / r["total_rows"] * 100).round(1),
            })
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 No missing values found!")

    # ── Outlier details ───────────────────────────────────────────────────────
    if r["outlier_details"]:
        st.markdown("#### ⚠️ Outlier Detection (3σ rule)")
        outlier_df = pd.DataFrame([
            {"Column": col, "Outlier Rows": count}
            for col, count in r["outlier_details"].items()
        ])
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)

    # ── Consistency issues ────────────────────────────────────────────────────
    if r["inconsistency_details"]:
        st.markdown("#### 🔀 Consistency Issues")
        for col, msg in r["inconsistency_details"].items():
            st.warning(f"**{col}**: {msg}")

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 Recommendations")
    recs = []
    if r["completeness"] < 95:
        recs.append(f"🔧 **{r['missing_cells']:,} missing cells** detected — consider imputation or row removal in the Cleaning step.")
    if r["duplicate_rate"] > 1:
        recs.append(f"🔧 **{r['duplicate_rows']} duplicate rows** found — remove them in the Cleaning step.")
    if r["outlier_details"]:
        recs.append(f"🔧 **Outliers** found in {len(r['outlier_details'])} column(s) — inspect and decide whether to cap or remove.")
    if r["inconsistency_details"]:
        recs.append(f"🔧 **Type inconsistencies** in {len(r['inconsistency_details'])} column(s) — consider casting to correct types.")
    if not recs:
        recs.append("✅ Your data looks clean! Proceed to Statistical Analysis.")

    for rec in recs:
        st.markdown(rec)

    st.markdown("---")
    st.markdown("""
    <div style="background:#1e1e2e;border:1px solid #2a2a4e;border-radius:10px;padding:1rem;">
        <p style="color:#94a3b8;margin:0;font-size:0.88rem;">
            ➡️ Next step: head to <strong style="color:#a78bfa;">🧹 Data Cleaning</strong> to fix the issues above.
        </p>
    </div>
    """, unsafe_allow_html=True)
