import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error
)
from utils.session import get_df, is_data_loaded

# Try to import optional libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def render():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: var(--text-primary); margin-bottom: 0.25rem;">🤖 Modeling & Evaluation Agent</h2>
        <p style="color: var(--text-secondary);">Train machine learning models on your dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        if st.button("← Back to Home"):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()
    
    # Initialize log
    if st.session_state.get("modeling_log") is None:
        st.session_state.modeling_log = []

    # Get column listings
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(all_cols) < 2:
        st.error("❌ The dataset must have at least 2 columns to perform modeling.")
        return

    # ── Pipeline Configuration ────────────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline Configuration")
    
    col_target, col_features = st.columns([1, 2])
    
    with col_target:
        st.markdown("#### 🎯 Target Selection")
        
        target_col = st.selectbox(
            "Select Target Variable (Y)",
            all_cols,
            help="The column you want the model to predict."
        )
        
        # Auto-detect task type
        unique_count = df[target_col].nunique()
        is_numeric_target = target_col in numeric_cols
        detected_task = "regression" if (is_numeric_target and unique_count > 12) else "classification"
        
        task_type = st.radio(
            "Task Type",
            ["classification", "regression"],
            index=0 if detected_task == "classification" else 1,
            format_func=lambda x: "🏷️ Classification" if x == "classification" else "📈 Regression"
        )

    with col_features:
        st.markdown("#### 📋 Feature Selection")
        
        default_features = [col for col in all_cols if col != target_col]
        select_all = st.checkbox("Select All Features", value=True)
        
        if select_all:
            features_list = default_features
            st.info(f"Using all {len(features_list)} remaining columns as features.")
        else:
            features_list = st.multiselect(
                "Select Features (X)",
                [c for c in all_cols if c != target_col],
                default=default_features[:min(8, len(default_features))]
            )

    st.markdown("---")

    # ── Training Settings ─────────────────────────────────────────────────────
    st.markdown("### 🛠️ Training Settings")
    
    col_split, col_scale, col_imbalance = st.columns(3)
    
    with col_split:
        test_size = st.slider("Test Split Size (%)", 10, 40, 20, 5) / 100.0
        random_state = st.number_input("Random Seed", value=42, step=1)
        cv_folds = st.number_input("Cross-Validation Folds", min_value=2, max_value=10, value=5)
        
    with col_scale:
        scaling_method = st.selectbox(
            "Feature Scaling",
            ["StandardScaler", "MinMaxScaler", "None"],
            index=0
        )
        
    with col_imbalance:
        if task_type == "classification" and SMOTE_AVAILABLE:
            handle_imbalance = st.checkbox("Handle Class Imbalance (SMOTE)", value=False)
        else:
            handle_imbalance = False
            if task_type == "classification" and not SMOTE_AVAILABLE:
                st.info("Install imbalanced-learn for SMOTE support")

    st.markdown("---")

    # ── Model Selection ─────────────────────────────────────────────────────
    st.markdown("### 🤖 Model Selection")
    
    if task_type == "classification":
        available_models = {}
        available_models["Random Forest"] = "rf"
        available_models["Gradient Boosting"] = "gbm"
        available_models["Logistic Regression"] = "logistic"
        available_models["Decision Tree"] = "dt"
        if XGB_AVAILABLE:
            available_models["XGBoost"] = "xgb"
        if LGB_AVAILABLE:
            available_models["LightGBM"] = "lgbm"
        
        default_selection = ["Random Forest"]
        if XGB_AVAILABLE:
            default_selection.append("XGBoost")
    else:
        available_models = {}
        available_models["Random Forest"] = "rf"
        available_models["Gradient Boosting"] = "gbm"
        available_models["Linear Regression"] = "linear"
        available_models["Ridge Regression"] = "ridge"
        if XGB_AVAILABLE:
            available_models["XGBoost"] = "xgb"
        if LGB_AVAILABLE:
            available_models["LightGBM"] = "lgbm"
        
        default_selection = ["Random Forest"]
        if XGB_AVAILABLE:
            default_selection.append("XGBoost")
        
    selected_model_names = st.multiselect(
        "Select Algorithms to Train",
        list(available_models.keys()),
        default=[m for m in default_selection if m in available_models]
    )

    if len(features_list) == 0:
        st.warning("⚠️ Please select at least one feature column to train models.")
        return
        
    if len(selected_model_names) == 0:
        st.warning("⚠️ Please select at least one algorithm to train.")
        return

    # ── Train button ──────────────────────────────────────────────────────────
    train_clicked = st.button("🚀 Run Modeling & Evaluation", use_container_width=True, type="primary")

    if train_clicked:
        run_training(df, target_col, features_list, task_type, selected_model_names, available_models, 
                     test_size, random_state, scaling_method, cv_folds, handle_imbalance)

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.get("modeling_done"):
        display_results(task_type)


def run_training(df, target_col, features_list, task_type, selected_model_names, available_models, 
                 test_size, random_state, scaling_method, cv_folds, handle_imbalance):
    
    st.session_state.modeling_log = []
    
    def log_step(msg):
        t = time.strftime("%H:%M:%S")
        st.session_state.modeling_log.append(f"[{t}] {msg}")
        
    log_step(f"Starting modeling for target: '{target_col}' ({task_type})")
    
    # 1. Data preparation
    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[features_list].copy()
    y = df_clean[target_col].copy()
    
    log_step(f"Data shape: {X.shape[0]} rows, {X.shape[1]} features")
    
    # 2. Encode categorical features
    cat_mappings = {}
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].fillna("Missing").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        cat_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        log_step(f"Encoded categorical column: {col}")
    
    # 3. Handle target encoding
    y_mapping = None
    y_inverse_mapping = None
    if task_type == "classification" and y.dtype == "object":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        y_inverse_mapping = {i: label for i, label in enumerate(le_y.classes_)}
        log_step(f"Encoded target with {len(le_y.classes_)} classes")
    
    # 4. Train-test split
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    log_step(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")
    
    # 5. Feature scaling
    scaler = None
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    if scaling_method != "None" and len(numeric_cols) > 0:
        log_step(f"Applying {scaling_method} scaling...")
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 6. SMOTE for imbalance
    if handle_imbalance and task_type == "classification" and SMOTE_AVAILABLE:
        log_step("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        log_step(f"After SMOTE: {len(X_train)} rows")
    
    # 7. Train models
    metrics_report = {}
    trained_models = {}
    
    console_ph = st.empty()
    
    for model_name in selected_model_names:
        log_step(f"Training {model_name}...")
        
        # Get model instance
        model = get_model_instance(model_name, task_type, random_state)
        
        if model is None:
            log_step(f"⚠️ {model_name} not available, skipping...")
            continue
        
        # Fit model
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        model_metrics = calculate_metrics(y_test, y_pred, task_type, model, X_test)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, 3), 
                                        scoring='accuracy' if task_type == 'classification' else 'r2')
            model_metrics["cv_mean"] = cv_scores.mean()
            model_metrics["cv_std"] = cv_scores.std()
        except:
            model_metrics["cv_mean"] = None
            model_metrics["cv_std"] = None
        
        metrics_report[model_name] = model_metrics
        log_step(f"✓ {model_name} complete")
        
        # Show progress
        with console_ph.container():
            st.markdown("#### 🪵 Training Log")
            st.code("\n".join(st.session_state.modeling_log[-10:]), language="text")
    
    # 8. Find best model
    if metrics_report:
        metric_key = "Accuracy" if task_type == "classification" else "R2 Score"
        best_model_name = max(metrics_report.items(), key=lambda x: x[1].get(metric_key, -np.inf))[0]
        best_model = trained_models[best_model_name]
        
        st.session_state.model_metrics = metrics_report
        st.session_state.trained_model = best_model
        st.session_state.trained_model_name = best_model_name
        st.session_state.model_features_list = features_list
        st.session_state.model_target_col = target_col
        st.session_state.model_scaler = scaler
        st.session_state.model_task_type = task_type
        st.session_state.model_encoded_categories = cat_mappings
        st.session_state.model_target_inverse_mapping = y_inverse_mapping
        st.session_state.modeling_done = True
        
        log_step(f"🏆 Best model: {best_model_name}")
        st.success(f"✅ Modeling complete! Best model: {best_model_name}")
        st.rerun()
    else:
        st.error("No models were successfully trained.")


def get_model_instance(model_name, task_type, random_state):
    """Get model instance with default parameters."""
    try:
        if task_type == "classification":
            if model_name == "Random Forest":
                return RandomForestClassifier(random_state=random_state, n_jobs=-1)
            elif model_name == "Gradient Boosting":
                return GradientBoostingClassifier(random_state=random_state)
            elif model_name == "Logistic Regression":
                return LogisticRegression(random_state=random_state, max_iter=1000)
            elif model_name == "Decision Tree":
                return DecisionTreeClassifier(random_state=random_state)
            elif model_name == "XGBoost" and XGB_AVAILABLE:
                return xgb.XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False)
            elif model_name == "LightGBM" and LGB_AVAILABLE:
                return lgb.LGBMClassifier(random_state=random_state, verbose=-1)
        else:
            if model_name == "Random Forest":
                return RandomForestRegressor(random_state=random_state, n_jobs=-1)
            elif model_name == "Gradient Boosting":
                return GradientBoostingRegressor(random_state=random_state)
            elif model_name == "Linear Regression":
                return LinearRegression()
            elif model_name == "Ridge Regression":
                return Ridge(random_state=random_state)
            elif model_name == "XGBoost" and XGB_AVAILABLE:
                return xgb.XGBRegressor(random_state=random_state)
            elif model_name == "LightGBM" and LGB_AVAILABLE:
                return lgb.LGBMRegressor(random_state=random_state, verbose=-1)
    except Exception as e:
        st.warning(f"Could not load {model_name}: {str(e)}")
        return None
    return None


def calculate_metrics(y_test, y_pred, task_type, model, X_test):
    """Calculate comprehensive metrics."""
    metrics = {}
    
    if task_type == "classification":
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["Precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        metrics["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        metrics["F1-Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
    else:
        metrics["R2 Score"] = r2_score(y_test, y_pred)
        metrics["MAE"] = mean_absolute_error(y_test, y_pred)
        metrics["MSE"] = mean_squared_error(y_test, y_pred)
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
    
    # Feature importance
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = model.feature_importances_.tolist()
    
    # Store predictions
    metrics["y_test"] = y_test.tolist()[:100]  # Limit for display
    metrics["y_pred"] = y_pred.tolist()[:100]
    
    return metrics


def display_results(task_type):
    """Display model results."""
    metrics = st.session_state.model_metrics
    best_model_name = st.session_state.trained_model_name
    
    st.markdown("---")
    st.markdown("### 🏆 Model Comparison")
    
    # Build comparison dataframe
    comp_data = []
    for model_name, data in metrics.items():
        row = {"Model": model_name}
        for metric, val in data.items():
            if isinstance(val, (int, float)):
                row[metric] = round(val, 4)
        comp_data.append(row)
    
    if comp_data:
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.success(f"💡 **Best Model:** {best_model_name}")
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Visualization", use_container_width=True):
            st.session_state.current_page = "visualization"
            st.rerun()
    with col2:
        if st.button("➡️ Proceed to AI Insights", use_container_width=True):
            st.session_state.current_page = "insights"
            st.rerun()
