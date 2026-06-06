import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from utils.session import get_df, is_data_loaded


def render():
    st.markdown("## 📊 Statistical Analysis Agent")
    st.markdown("Descriptive statistics, correlations, and distribution analysis — fully local, no AI cost.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if st.button("▶ Run Statistical Analysis", use_container_width=False):
        st.session_state.stats_done = True

    if not st.session_state.get("stats_done"):
        st.info("Click **Run Statistical Analysis** to generate the report.")
        return

    # ── Tabs inside the page ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📐 Descriptive", "🔗 Correlations", "📦 Distributions", "🏷️ Categorical"])

    # ── Tab 1: Descriptive stats ──────────────────────────────────────────────
    with tab1:
        st.markdown("#### Descriptive Statistics (Numeric Columns)")
        if numeric_cols:
            desc = df[numeric_cols].describe().T
            desc["skewness"] = df[numeric_cols].skew()
            desc["kurtosis"] = df[numeric_cols].kurtosis()
            desc["missing"] = df[numeric_cols].isnull().sum()
            desc["missing_%"] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
            st.dataframe(desc.style.format("{:.3f}", na_rep="—"), use_container_width=True)

            # Quick interpretation
            st.markdown("#### 📌 Key Observations")
            for col in numeric_cols:
                skew = df[col].skew()
                if abs(skew) > 1:
                    direction = "right (positive)" if skew > 0 else "left (negative)"
                    st.markdown(f"- **{col}** is strongly skewed {direction} (skew={skew:.2f})")
        else:
            st.info("No numeric columns found in this dataset.")

    # ── Tab 2: Correlations ───────────────────────────────────────────────────
    with tab2:
        if len(numeric_cols) >= 2:
            st.markdown("#### Pearson Correlation Matrix")
            corr = df[numeric_cols].corr()
            st.dataframe(
                corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.3f}"),
                use_container_width=True,
            )

            # Top correlations
            st.markdown("#### 🔝 Top Correlated Pairs")
            pairs = []
            cols_list = corr.columns.tolist()
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    pairs.append({
                        "Variable A": cols_list[i],
                        "Variable B": cols_list[j],
                        "Correlation": round(corr.iloc[i, j], 4),
                        "Strength": _corr_strength(corr.iloc[i, j]),
                    })
            pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
            st.dataframe(pairs_df.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # ── Tab 3: Distributions ──────────────────────────────────────────────────
    with tab3:
        if numeric_cols:
            st.markdown("#### Distribution Summary")
            selected_col = st.selectbox("Select column to inspect", numeric_cols)
            col_data = df[selected_col].dropna()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{col_data.mean():.3f}")
            c2.metric("Median", f"{col_data.median():.3f}")
            c3.metric("Std Dev", f"{col_data.std():.3f}")
            c4.metric("IQR", f"{col_data.quantile(0.75) - col_data.quantile(0.25):.3f}")

            # Normality test
            if len(col_data) >= 8:
                stat, p_value = stats.shapiro(col_data.sample(min(500, len(col_data)), random_state=42))
                normal = p_value > 0.05
                st.markdown(
                    f"**Shapiro-Wilk Normality Test:** stat={stat:.4f}, p={p_value:.4f} — "
                    + ("✅ Likely normal" if normal else "❌ Not normally distributed")
                )

            # Percentile table
            percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            perc_data = {f"P{int(p*100)}": col_data.quantile(p) for p in percentiles}
            st.dataframe(pd.DataFrame(perc_data, index=[selected_col]).T.rename(columns={selected_col: "Value"}),
                         use_container_width=True)
        else:
            st.info("No numeric columns available.")

    # ── Tab 4: Categorical ────────────────────────────────────────────────────
    with tab4:
        if cat_cols:
            st.markdown("#### Categorical Column Summary")
            selected_cat = st.selectbox("Select categorical column", cat_cols)
            col_data = df[selected_cat]

            c1, c2, c3 = st.columns(3)
            c1.metric("Unique Values", col_data.nunique())
            c2.metric("Most Common", str(col_data.mode().iloc[0]) if not col_data.mode().empty else "—")
            c3.metric("Missing", col_data.isnull().sum())

            freq_df = col_data.value_counts().reset_index()
            freq_df.columns = ["Value", "Count"]
            freq_df["Percentage"] = (freq_df["Count"] / len(df) * 100).round(2)
            st.dataframe(freq_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns found.")

    # Store stats for report
    if numeric_cols:
        st.session_state.stats_report = {
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
            "describe": df[numeric_cols].describe().to_dict() if numeric_cols else {},
        }

    st.markdown("---")
    st.markdown("""
    <div style="background:#1e1e2e;border:1px solid #2a2a4e;border-radius:10px;padding:1rem;">
        <p style="color:#94a3b8;margin:0;font-size:0.88rem;">
            ➡️ Next step: head to <strong style="color:#a78bfa;">📈 Visualization</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)


def _corr_strength(val: float) -> str:
    a = abs(val)
    if a >= 0.8:
        return "🔴 Very strong"
    elif a >= 0.6:
        return "🟠 Strong"
    elif a >= 0.4:
        return "🟡 Moderate"
    elif a >= 0.2:
        return "🔵 Weak"
    else:
        return "⚪ Negligible"
