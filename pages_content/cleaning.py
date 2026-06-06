import streamlit as st
import pandas as pd
import numpy as np
from utils.session import get_df, get_raw_df, update_clean_df, is_data_loaded
import io


def render():
    st.markdown("## 🧹 Data Cleaning Agent")
    st.markdown("Automatically detects and fixes common data issues. All operations are transparent and reversible.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df_raw = get_raw_df()
    df = get_df()

    # ── Cleaning options ──────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Cleaning Options")

    col1, col2 = st.columns(2)
    with col1:
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        fix_missing_numeric = st.selectbox(
            "Handle missing numeric values",
            ["Fill with median", "Fill with mean", "Fill with 0", "Drop rows"],
            index=0,
        )
        fix_missing_categorical = st.selectbox(
            "Handle missing categorical values",
            ["Fill with mode", "Fill with 'Unknown'", "Drop rows"],
            index=0,
        )
    with col2:
        strip_whitespace = st.checkbox("Strip whitespace from text columns", value=True)
        fix_column_names = st.checkbox("Standardize column names (lowercase, underscores)", value=True)
        drop_all_null_cols = st.checkbox("Drop columns that are entirely empty", value=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶ Run Data Cleaning", use_container_width=False):
        df_clean = df_raw.copy()
        log = []

        # 1. Standardize column names
        if fix_column_names:
            old_cols = df_clean.columns.tolist()
            df_clean.columns = (
                df_clean.columns
                .str.strip()
                .str.lower()
                .str.replace(r"[\s\-\.]+", "_", regex=True)
                .str.replace(r"[^\w]", "", regex=True)
            )
            changed = [(o, n) for o, n in zip(old_cols, df_clean.columns) if o != n]
            if changed:
                log.append(("✅", "Column names standardized", f"{len(changed)} column(s) renamed"))

        # 2. Drop all-null columns
        if drop_all_null_cols:
            null_cols = df_clean.columns[df_clean.isnull().all()].tolist()
            if null_cols:
                df_clean.drop(columns=null_cols, inplace=True)
                log.append(("✅", "Empty columns removed", f"Dropped: {', '.join(null_cols)}"))

        # 3. Strip whitespace
        if strip_whitespace:
            obj_cols = df_clean.select_dtypes(include="object").columns
            for col in obj_cols:
                df_clean[col] = df_clean[col].str.strip()
            if len(obj_cols) > 0:
                log.append(("✅", "Whitespace stripped", f"Applied to {len(obj_cols)} text column(s)"))

        # 4. Remove duplicates
        if remove_duplicates:
            before = len(df_clean)
            df_clean.drop_duplicates(inplace=True)
            removed = before - len(df_clean)
            if removed > 0:
                log.append(("✅", "Duplicate rows removed", f"{removed} duplicate row(s) dropped"))
            else:
                log.append(("ℹ️", "No duplicates found", "Dataset was already clean"))

        # 5. Fix missing numeric
        numeric_cols = df_clean.select_dtypes(include="number").columns
        missing_numeric = df_clean[numeric_cols].isnull().sum()
        missing_numeric = missing_numeric[missing_numeric > 0]
        if len(missing_numeric) > 0:
            if fix_missing_numeric == "Fill with median":
                for col in missing_numeric.index:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                log.append(("✅", "Numeric missing values filled", f"Used median for {list(missing_numeric.index)}"))
            elif fix_missing_numeric == "Fill with mean":
                for col in missing_numeric.index:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                log.append(("✅", "Numeric missing values filled", f"Used mean for {list(missing_numeric.index)}"))
            elif fix_missing_numeric == "Fill with 0":
                for col in missing_numeric.index:
                    df_clean[col].fillna(0, inplace=True)
                log.append(("✅", "Numeric missing values filled", f"Filled with 0 for {list(missing_numeric.index)}"))
            elif fix_missing_numeric == "Drop rows":
                before = len(df_clean)
                df_clean.dropna(subset=missing_numeric.index, inplace=True)
                log.append(("✅", "Rows with missing numeric values dropped", f"{before - len(df_clean)} row(s) removed"))

        # 6. Fix missing categorical
        cat_cols = df_clean.select_dtypes(include="object").columns
        missing_cat = df_clean[cat_cols].isnull().sum()
        missing_cat = missing_cat[missing_cat > 0]
        if len(missing_cat) > 0:
            if fix_missing_categorical == "Fill with mode":
                for col in missing_cat.index:
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col].fillna(mode_val[0], inplace=True)
                log.append(("✅", "Categorical missing values filled", f"Used mode for {list(missing_cat.index)}"))
            elif fix_missing_categorical == "Fill with 'Unknown'":
                for col in missing_cat.index:
                    df_clean[col].fillna("Unknown", inplace=True)
                log.append(("✅", "Categorical missing values filled", f"Used 'Unknown' for {list(missing_cat.index)}"))
            elif fix_missing_categorical == "Drop rows":
                before = len(df_clean)
                df_clean.dropna(subset=missing_cat.index, inplace=True)
                log.append(("✅", "Rows with missing categorical values dropped", f"{before - len(df_clean)} row(s) removed"))

        # Save to session
        update_clean_df(df_clean)
        st.session_state.cleaning_report = {
            "log": log,
            "before_shape": df_raw.shape,
            "after_shape": df_clean.shape,
        }
        st.success("✅ Cleaning complete!")

    # ── Show results ──────────────────────────────────────────────────────────
    if st.session_state.get("cleaning_report"):
        r = st.session_state.cleaning_report
        log = r["log"]

        st.markdown("---")
        st.markdown("#### 📋 Cleaning Log")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows Before", f"{r['before_shape'][0]:,}")
        col2.metric("Rows After", f"{r['after_shape'][0]:,}")
        col3.metric("Rows Removed", f"{r['before_shape'][0] - r['after_shape'][0]:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        for icon, action, detail in log:
            st.markdown(f"""
            <div style="background:#1e1e2e;border-left:3px solid #6c63ff;border-radius:6px;
                        padding:0.6rem 1rem;margin-bottom:0.5rem;">
                <span style="color:#a78bfa;font-weight:600;">{icon} {action}</span>
                <span style="color:#64748b;font-size:0.85rem;margin-left:0.8rem;">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 👀 Cleaned Data Preview")
        df_clean = get_df()
        st.dataframe(df_clean.head(15), use_container_width=True)

        # Download button
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("""
        <div style="background:#1e1e2e;border:1px solid #2a2a4e;border-radius:10px;padding:1rem;">
            <p style="color:#94a3b8;margin:0;font-size:0.88rem;">
                ➡️ Next step: head to <strong style="color:#a78bfa;">📊 Statistical Analysis</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Configure the options above and click **Run Data Cleaning** to begin.")
