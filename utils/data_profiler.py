import pandas as pd
import numpy as np
from scipy import stats


def generate_data_profile(df: pd.DataFrame) -> dict:
    """Generate comprehensive data profile report."""
    
    profile = {
        "basic_info": {},
        "numeric_stats": {},
        "categorical_stats": {},
        "missing_analysis": {},
        "correlation_analysis": {},
        "outlier_analysis": {},
        "data_quality": {}
    }
    
    # Basic Information
    profile["basic_info"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_percentage": round(df.duplicated().sum() / len(df) * 100, 2)
    }
    
    # Numeric Columns Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            profile["numeric_stats"][col] = {
                "count": len(data),
                "missing": df[col].isnull().sum(),
                "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2),
                "mean": round(data.mean(), 4),
                "median": round(data.median(), 4),
                "std": round(data.std(), 4),
                "var": round(data.var(), 4),
                "min": round(data.min(), 4),
                "max": round(data.max(), 4),
                "range": round(data.max() - data.min(), 4),
                "q1": round(data.quantile(0.25), 4),
                "q3": round(data.quantile(0.75), 4),
                "iqr": round(data.quantile(0.75) - data.quantile(0.25), 4),
                "skewness": round(data.skew(), 4),
                "kurtosis": round(data.kurtosis(), 4),
                "unique_values": data.nunique(),
                "zero_count": (data == 0).sum(),
                "negative_count": (data < 0).sum(),
                "positive_count": (data > 0).sum()
            }
    
    # Categorical Columns Analysis
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        data = df[col].dropna()
        if len(data) > 0:
            value_counts = data.value_counts()
            profile["categorical_stats"][col] = {
                "count": len(data),
                "missing": df[col].isnull().sum(),
                "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2),
                "unique_values": data.nunique(),
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "most_frequent_pct": round(value_counts.iloc[0] / len(data) * 100, 2) if len(value_counts) > 0 else 0,
                "top_5_categories": {str(k): int(v) for k, v in value_counts.head(5).items()}
            }
    
    # Missing Value Analysis
    missing_df = df.isnull().sum()
    missing_cols = missing_df[missing_df > 0]
    profile["missing_analysis"] = {
        "total_missing_cells": int(missing_df.sum()),
        "total_missing_pct": round(missing_df.sum() / df.size * 100, 2),
        "columns_with_missing": len(missing_cols),
        "missing_by_column": {col: {"count": int(count), "percentage": round(count / len(df) * 100, 2)} 
                              for col, count in missing_cols.items()}
    }
    
    # Correlation Analysis (for numeric columns)
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        # Get top correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlations.append({
                    "variable_1": corr_matrix.columns[i],
                    "variable_2": corr_matrix.columns[j],
                    "correlation": round(corr_matrix.iloc[i, j], 4),
                    "strength": "Strong" if abs(corr_matrix.iloc[i, j]) >= 0.7 else "Moderate" if abs(corr_matrix.iloc[i, j]) >= 0.4 else "Weak"
                })
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        profile["correlation_analysis"] = {
            "matrix": corr_matrix.to_dict(),
            "top_10_correlations": correlations[:10]
        }
    
    # Outlier Analysis (using IQR method)
    outlier_summary = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            outlier_summary[col] = {
                "outlier_count": len(outliers),
                "outlier_percentage": round(len(outliers) / len(data) * 100, 2),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "min_outlier": round(outliers.min(), 4) if len(outliers) > 0 else None,
                "max_outlier": round(outliers.max(), 4) if len(outliers) > 0 else None
            }
    profile["outlier_analysis"] = outlier_summary
    
    # Data Quality Score
    completeness_score = (1 - missing_df.sum() / df.size) * 100
    uniqueness_score = (1 - df.duplicated().sum() / len(df)) * 100
    outlier_penalty = sum([v["outlier_percentage"] for v in outlier_summary.values()]) / len(outlier_summary) if outlier_summary else 0
    
    profile["data_quality"] = {
        "completeness_score": round(completeness_score, 2),
        "uniqueness_score": round(uniqueness_score, 2),
        "overall_quality_score": round((completeness_score + uniqueness_score - outlier_penalty) / 2, 2),
        "quality_grade": "Excellent" if completeness_score > 95 else "Good" if completeness_score > 80 else "Fair" if completeness_score > 60 else "Poor"
    }
    
    return profile


def display_data_profile(profile: dict):
    """Display data profile in a nice format."""
    
    # Basic Info
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{profile['basic_info']['total_rows']:,}")
    with col2:
        st.metric("Total Columns", profile['basic_info']['total_columns'])
    with col3:
        st.metric("Memory Usage", f"{profile['basic_info']['memory_usage_mb']:.2f} MB")
    with col4:
        st.metric("Quality Grade", profile['data_quality']['quality_grade'])
    
    # Quality Scores
    st.markdown("### ✅ Data Quality Scores")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completeness", f"{profile['data_quality']['completeness_score']:.1f}%")
    with col2:
        st.metric("Uniqueness", f"{profile['data_quality']['uniqueness_score']:.1f}%")
    with col3:
        st.metric("Overall Score", f"{profile['data_quality']['overall_quality_score']:.1f}/100")
    
    # Missing Values
    if profile['missing_analysis']['columns_with_missing'] > 0:
        st.markdown("### ⚠️ Missing Values Analysis")
        missing_df = pd.DataFrame([
            {"Column": col, "Missing Count": info['count'], "Missing %": info['percentage']}
            for col, info in profile['missing_analysis']['missing_by_column'].items()
        ]).sort_values("Missing %", ascending=False)
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Numeric Statistics
    if profile['numeric_stats']:
        st.markdown("### 📊 Numeric Columns Statistics")
        stats_data = []
        for col, stats_dict in profile['numeric_stats'].items():
            stats_data.append({
                "Column": col,
                "Mean": stats_dict['mean'],
                "Median": stats_dict['median'],
                "Std Dev": stats_dict['std'],
                "Min": stats_dict['min'],
                "Max": stats_dict['max'],
                "Skewness": stats_dict['skewness'],
                "Missing %": stats_dict['missing_pct']
            })
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
    # Outlier Analysis
    if profile['outlier_analysis']:
        st.markdown("### 🔍 Outlier Analysis")
        outlier_data = []
        for col, info in profile['outlier_analysis'].items():
            if info['outlier_count'] > 0:
                outlier_data.append({
                    "Column": col,
                    "Outlier Count": info['outlier_count'],
                    "Outlier %": f"{info['outlier_percentage']:.1f}%",
                    "Lower Bound": info['lower_bound'],
                    "Upper Bound": info['upper_bound']
                })
        if outlier_data:
            st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)
        else:
            st.info("No significant outliers detected!")
    
    # Top Correlations
    if profile['correlation_analysis'] and profile['correlation_analysis'].get('top_10_correlations'):
        st.markdown("### 🔗 Top Correlations")
        corr_df = pd.DataFrame(profile['correlation_analysis']['top_10_correlations'])
        st.dataframe(corr_df, use_container_width=True)
    
    # Categorical Summary
    if profile['categorical_stats']:
        st.markdown("### 🏷️ Categorical Columns Summary")
        for col, stats_dict in list(profile['categorical_stats'].items())[:5]:
            with st.expander(f"📌 {col}"):
                st.write(f"**Unique Values:** {stats_dict['unique_values']}")
                st.write(f"**Most Frequent:** {stats_dict['most_frequent']} ({stats_dict['most_frequent_count']} rows, {stats_dict['most_frequent_pct']}%)")
                st.write(f"**Missing:** {stats_dict['missing']} ({stats_dict['missing_pct']}%)")
                if stats_dict['top_5_categories']:
                    st.write("**Top Categories:**")
                    for cat, count in list(stats_dict['top_5_categories'].items())[:5]:
                        st.write(f"  - {cat}: {count} rows")