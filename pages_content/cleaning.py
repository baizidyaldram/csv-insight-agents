import streamlit as st
import pandas as pd
import numpy as np
import io  
from utils.session import get_df, get_raw_df, update_clean_df, is_data_loaded


def render():
    st.markdown("## 🧹 Data Cleaning Agent")
    st.markdown("Automatically detects and fixes common data issues with full transparency.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded.")
        # Add back button
        if st.button("← Back to Home", use_container_width=False):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df_raw = get_raw_df()
    
    # Show current data statistics
    st.markdown("### 📊 Current Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df_raw.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df_raw.shape[1])
    with col3:
        duplicates = df_raw.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicates:,}", delta=f"{duplicates/len(df_raw)*100:.1f}%")
    with col4:
        missing = df_raw.isnull().sum().sum()
        st.metric("Missing Values", f"{missing:,}", delta=f"{missing/df_raw.size*100:.1f}%")
    
    st.markdown("---")
    
    # Cleaning options with better defaults
    st.markdown("### ⚙️ Cleaning Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗑️ Row Operations")
        remove_duplicates = st.checkbox(
            "Remove duplicate rows", 
            value=True,
            help="Removes exact duplicate rows from the dataset"
        )
        
        remove_missing_rows = st.checkbox(
            "Remove rows with missing values", 
            value=False,
            help="⚠️ WARNING: This will remove any row with at least one missing value"
        )
        
        missing_threshold = st.slider(
            "Max missing values per row (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=10,
            help="Remove rows that have more than X% missing values"
        ) if not remove_missing_rows else 100
        
    with col2:
        st.markdown("#### 🔧 Column Operations")
        fill_missing_numeric = st.selectbox(
            "Fill missing numeric values with",
            ["None (keep as is)", "Median", "Mean", "Zero", "Forward Fill", "Backward Fill"],
            index=0,
            help="Strategy for handling missing numeric values"
        )
        
        fill_missing_categorical = st.selectbox(
            "Fill missing categorical values with",
            ["None (keep as is)", "Mode (most frequent)", "Unknown", "Forward Fill", "Backward Fill"],
            index=0,
            help="Strategy for handling missing categorical values"
        )
        
        standardize_column_names = st.checkbox(
            "Standardize column names",
            value=False,
            help="Convert column names to lowercase with underscores"
        )
    
    st.markdown("---")
    
    # Advanced options (collapsible)
    with st.expander("🔧 Advanced Options (Optional)"):
        col1, col2 = st.columns(2)
        
        with col1:
            handle_outliers = st.checkbox(
                "Handle numeric outliers",
                value=False,
                help="Cap outliers at 3 standard deviations"
            )
            
            outlier_method = st.selectbox(
                "Outlier handling method",
                ["Cap at 3 std dev", "Cap at IQR", "Remove outliers"],
                disabled=not handle_outliers
            )
        
        with col2:
            remove_constant_columns = st.checkbox(
                "Remove constant columns",
                value=False,
                help="Remove columns with only one unique value"
            )
            
            remove_high_cardinality = st.checkbox(
                "Remove high cardinality columns",
                value=False,
                help="Remove categorical columns with too many unique values"
            )
            
            cardinality_threshold = st.slider(
                "Max unique values threshold",
                min_value=10,
                max_value=500,
                value=100,
                step=50,
                disabled=not remove_high_cardinality
            )
    
    st.markdown("---")
    
    # Preview changes before applying
    if st.button("🔍 Preview Changes", use_container_width=False):
        df_preview = df_raw.copy()
        changes = []
        
        # Simulate changes for preview
        before_rows = len(df_preview)
        
        if remove_duplicates:
            dup_count = df_preview.duplicated().sum()
            df_preview.drop_duplicates(inplace=True)
            if dup_count > 0:
                changes.append(f"• Will remove {dup_count:,} duplicate rows")
        
        if remove_missing_rows:
            before = len(df_preview)
            df_preview.dropna(inplace=True)
            removed = before - len(df_preview)
            if removed > 0:
                changes.append(f"• Will remove {removed:,} rows with missing values")
        elif missing_threshold < 100:
            max_missing = int(len(df_preview.columns) * missing_threshold / 100)
            before = len(df_preview)
            df_preview = df_preview[df_preview.isnull().sum(axis=1) <= max_missing]
            removed = before - len(df_preview)
            if removed > 0:
                changes.append(f"• Will remove {removed:,} rows with >{missing_threshold}% missing values")
        
        if fill_missing_numeric != "None (keep as is)":
            numeric_cols = df_preview.select_dtypes(include="number").columns
            missing_before = df_preview[numeric_cols].isnull().sum().sum()
            changes.append(f"• Will fill {missing_before:,} missing numeric values using '{fill_missing_numeric}'")
        
        if fill_missing_categorical != "None (keep as is)":
            cat_cols = df_preview.select_dtypes(include="object").columns
            missing_before = df_preview[cat_cols].isnull().sum().sum()
            changes.append(f"• Will fill {missing_before:,} missing categorical values using '{fill_missing_categorical}'")
        
        if changes:
            st.info("### 📋 Preview of Changes:")
            for change in changes:
                st.write(change)
            st.warning(f"⚠️ Preview shows {before_rows - len(df_preview):,} rows will be removed ({((before_rows - len(df_preview))/before_rows*100):.1f}% of data)")
        else:
            st.info("No changes will be applied with current settings")
    
    # Run cleaning button
    if st.button("▶ Run Data Cleaning", use_container_width=False, type="primary"):
        df_clean = df_raw.copy()
        log = []
        rows_before = len(df_clean)
        
        # 1. Remove duplicates
        if remove_duplicates:
            dup_count = df_clean.duplicated().sum()
            if dup_count > 0:
                df_clean.drop_duplicates(inplace=True)
                log.append(f"✅ Removed {dup_count:,} duplicate rows")
        
        # 2. Handle missing values in rows
        if remove_missing_rows:
            before = len(df_clean)
            df_clean.dropna(inplace=True)
            removed = before - len(df_clean)
            if removed > 0:
                log.append(f"✅ Removed {removed:,} rows with missing values")
        elif missing_threshold < 100:
            max_missing = int(len(df_clean.columns) * missing_threshold / 100)
            before = len(df_clean)
            df_clean = df_clean[df_clean.isnull().sum(axis=1) <= max_missing]
            removed = before - len(df_clean)
            if removed > 0:
                log.append(f"✅ Removed {removed:,} rows with >{missing_threshold}% missing values")
        
        # 3. Fill missing numeric values
        if fill_missing_numeric != "None (keep as is)":
            numeric_cols = df_clean.select_dtypes(include="number").columns
            missing_before = df_clean[numeric_cols].isnull().sum().sum()
            
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    if fill_missing_numeric == "Median":
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif fill_missing_numeric == "Mean":
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif fill_missing_numeric == "Zero":
                        df_clean[col].fillna(0, inplace=True)
                    elif fill_missing_numeric == "Forward Fill":
                        df_clean[col].fillna(method='ffill', inplace=True)
                    elif fill_missing_numeric == "Backward Fill":
                        df_clean[col].fillna(method='bfill', inplace=True)
            
            missing_after = df_clean[numeric_cols].isnull().sum().sum()
            filled = missing_before - missing_after
            if filled > 0:
                log.append(f"✅ Filled {filled:,} missing numeric values using '{fill_missing_numeric}'")
        
        # 4. Fill missing categorical values
        if fill_missing_categorical != "None (keep as is)":
            cat_cols = df_clean.select_dtypes(include="object").columns
            missing_before = df_clean[cat_cols].isnull().sum().sum()
            
            for col in cat_cols:
                if df_clean[col].isnull().any():
                    if fill_missing_categorical == "Mode (most frequent)":
                        mode_val = df_clean[col].mode()
                        if len(mode_val) > 0:
                            df_clean[col].fillna(mode_val[0], inplace=True)
                    elif fill_missing_categorical == "Unknown":
                        df_clean[col].fillna("Unknown", inplace=True)
                    elif fill_missing_categorical == "Forward Fill":
                        df_clean[col].fillna(method='ffill', inplace=True)
                    elif fill_missing_categorical == "Backward Fill":
                        df_clean[col].fillna(method='bfill', inplace=True)
            
            missing_after = df_clean[cat_cols].isnull().sum().sum()
            filled = missing_before - missing_after
            if filled > 0:
                log.append(f"✅ Filled {filled:,} missing categorical values using '{fill_missing_categorical}'")
        
        # 5. Handle outliers (if selected)
        if handle_outliers:
            numeric_cols = df_clean.select_dtypes(include="number").columns
            outlier_count = 0
            
            for col in numeric_cols:
                if outlier_method == "Cap at 3 std dev":
                    mean = df_clean[col].mean()
                    std = df_clean[col].std()
                    upper = mean + 3 * std
                    lower = mean - 3 * std
                    before_outliers = ((df_clean[col] > upper) | (df_clean[col] < lower)).sum()
                    df_clean[col] = df_clean[col].clip(lower, upper)
                    outlier_count += before_outliers
                elif outlier_method == "Cap at IQR":
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    upper = Q3 + 1.5 * IQR
                    lower = Q1 - 1.5 * IQR
                    before_outliers = ((df_clean[col] > upper) | (df_clean[col] < lower)).sum()
                    df_clean[col] = df_clean[col].clip(lower, upper)
                    outlier_count += before_outliers
                elif outlier_method == "Remove outliers":
                    mean = df_clean[col].mean()
                    std = df_clean[col].std()
                    upper = mean + 3 * std
                    lower = mean - 3 * std
                    before = len(df_clean)
                    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
                    outlier_count += before - len(df_clean)
            
            if outlier_count > 0:
                log.append(f"✅ Handled {outlier_count:,} outliers using '{outlier_method}'")
        
        # 6. Remove constant columns
        if remove_constant_columns:
            constant_cols = []
            for col in df_clean.columns:
                if df_clean[col].nunique() == 1:
                    constant_cols.append(col)
            if constant_cols:
                df_clean.drop(columns=constant_cols, inplace=True)
                log.append(f"✅ Removed {len(constant_cols)} constant columns: {', '.join(constant_cols[:5])}")
        
        # 7. Remove high cardinality columns
        if remove_high_cardinality:
            high_card_cols = []
            cat_cols = df_clean.select_dtypes(include="object").columns
            for col in cat_cols:
                if df_clean[col].nunique() > cardinality_threshold:
                    high_card_cols.append(col)
            if high_card_cols:
                df_clean.drop(columns=high_card_cols, inplace=True)
                log.append(f"✅ Removed {len(high_card_cols)} high-cardinality columns (> {cardinality_threshold} unique values)")
        
        # 8. Standardize column names
        if standardize_column_names:
            original_cols = df_clean.columns.tolist()
            df_clean.columns = (df_clean.columns
                .str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('-', '_')
            )
            log.append(f"✅ Standardized {len(original_cols)} column names")
        
        # Save results
        update_clean_df(df_clean)
        rows_after = len(df_clean)
        rows_removed = rows_before - rows_after
        
        st.session_state.cleaning_report = {
            "log": log,
            "rows_removed": rows_removed,
            "before_shape": df_raw.shape,
            "after_shape": df_clean.shape,
        }
        
        if rows_removed > 0:
            st.warning(f"⚠️ Removed {rows_removed:,} rows ({rows_removed/rows_before*100:.1f}% of data)")
        st.success("✅ Cleaning complete!")
    
    # Display results
    if st.session_state.get("cleaning_report"):
        r = st.session_state.cleaning_report
        log = r.get("log", [])
        
        st.markdown("---")
        st.markdown("#### 📋 Cleaning Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Before", f"{r['before_shape'][0]:,}")
        with col2:
            st.metric("Rows After", f"{r['after_shape'][0]:,}")
        with col3:
            delta_pct = (r['rows_removed'] / r['before_shape'][0] * 100) if r['before_shape'][0] > 0 else 0
            st.metric("Rows Removed", f"{r['rows_removed']:,}", delta=f"{delta_pct:.1f}%")
        
        if log:
            st.markdown("#### 📝 Cleaning Log")
            for item in log:
                st.info(item)
        
        st.markdown("---")
        st.markdown("#### 👀 Cleaned Data Preview")
        df_clean = get_df()
        st.dataframe(df_clean.head(15), use_container_width=True)
        
        # Download cleaned data
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col_btn2:
            if st.button("➡️ Next: Statistical Analysis", use_container_width=True):
                st.session_state.current_page = "stats"
                st.rerun()
    else:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col_btn2:
            st.info("Configure cleaning options above and click **Run Data Cleaning** to begin.")
