import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error
)
from utils.session import get_df, is_data_loaded

def render():
    st.markdown("""
    <div style="background:linear-gradient(90deg, #6c63ff 0%, #3b82f6 100%); padding: 1px; border-radius: 12px; margin-bottom: 1.5rem;">
        <div style="background:#0a0915; padding: 1.5rem; border-radius: 11px;">
            <h2 style="margin:0 0 0.2rem 0; color:#e0e0f0; font-size:1.8rem;">🤖 Modeling & Evaluation Agent</h2>
            <p style="margin:0; color:#64748b; font-size:0.95rem;">
                Train machine learning models locally on your dataset, analyze their performance, and play with predictions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()
    
    # ── Sidebar Agent Console Logs ──────────────────────────────────────────────
    if st.session_state.get("modeling_log") is None:
        st.session_state.modeling_log = []

    # Get column listings
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(all_cols) < 2:
        st.error("❌ The dataset must have at least 2 columns to perform modeling.")
        return

    # ── Pipeline configurations ────────────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline Configuration")
    
    col_target, col_features = st.columns([1, 2])
    
    with col_target:
        st.markdown('<div class="agent-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("<h4 style='color:#a78bfa;'>🎯 Target Selection</h4>", unsafe_allow_html=True)
        
        target_col = st.selectbox(
            "Select Target Variable (Y)",
            all_cols,
            index=len(all_cols) - 1,
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
            format_func=lambda x: "🏷️ Classification" if x == "classification" else "📈 Regression",
            help="Classification for categories/labels, Regression for continuous numeric values."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_features:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color:#a78bfa;'>📋 Feature Selection</h4>", unsafe_allow_html=True)
        
        # Select all features by default except target
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
            
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training Settings ─────────────────────────────────────────────────────
    st.markdown("### 🛠️ Training Settings")
    c_split, c_scale, c_models = st.columns([1, 1, 2])
    
    with c_split:
        test_size = st.slider("Test Split Size (%)", 10, 50, 20, 5) / 100.0
        random_state = st.number_input("Random Seed (State)", value=42, step=1)
        
    with c_scale:
        scaling_method = st.selectbox(
            "Feature Scaling",
            ["None", "StandardScaler", "MinMaxScaler"],
            index=1 if len(numeric_cols) > 0 else 0
        )
        
    with c_models:
        if task_type == "classification":
            available_models = {
                "Random Forest": "rf",
                "Gradient Boosting": "gbm",
                "Logistic Regression": "logistic",
                "Decision Tree": "dt"
            }
            default_selection = ["Random Forest", "Logistic Regression"]
        else:
            available_models = {
                "Random Forest": "rf",
                "Gradient Boosting": "gbm",
                "Linear Regression": "linear",
                "Ridge Regression": "ridge"
            }
            default_selection = ["Random Forest", "Linear Regression"]
            
        selected_model_names = st.multiselect(
            "Select Algorithms to Train",
            list(available_models.keys()),
            default=[m for m in default_selection if m in available_models]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Train button ──────────────────────────────────────────────────────────
    if len(features_list) == 0:
        st.warning("⚠️ Please select at least one feature column to train models.")
        return
        
    if len(selected_model_names) == 0:
        st.warning("⚠️ Please select at least one algorithm to train.")
        return

    train_clicked = st.button("▶ Run Modeling & Evaluation", use_container_width=True)

    if train_clicked:
        run_training(df, target_col, features_list, task_type, selected_model_names, available_models, test_size, random_state, scaling_method)

    # ── Display logs & results ────────────────────────────────────────────────
    if st.session_state.get("modeling_done"):
        display_results(task_type)

def run_training(df, target_col, features_list, task_type, selected_model_names, available_models, test_size, random_state, scaling_method):
    st.session_state.modeling_log = []
    
    def log_step(msg):
        t = time.strftime("%H:%M:%S")
        st.session_state.modeling_log.append(f"[{t}] 🤖 {msg}")
        
    log_step(f"Starting pipeline setup for predicting '{target_col}' ({task_type})")
    
    # 1. Filter out missing targets
    initial_shape = df.shape
    df_clean = df.dropna(subset=[target_col]).copy()
    rows_dropped = initial_shape[0] - df_clean.shape[0]
    if rows_dropped > 0:
        log_step(f"Dropped {rows_dropped} rows due to missing target values.")
    
    # Split features and target
    X = df_clean[features_list].copy()
    y = df_clean[target_col].copy()
    
    # Identify feature types
    num_feats = X.select_dtypes(include="number").columns.tolist()
    cat_feats = X.select_dtypes(include="object").columns.tolist()
    log_step(f"Identified features: {len(num_feats)} numeric, {len(cat_feats)} categorical.")
    
    # 2. Impute features
    log_step("Imputing missing values...")
    # Numeric imputation
    for col in num_feats:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            
    # Categorical imputation & encoding
    cat_mappings = {}
    for col in cat_feats:
        # Fill missing with 'Missing'
        X[col] = X[col].fillna("Missing").astype(str)
        # Label encode
        unique_vals = sorted(X[col].unique().tolist())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        cat_mappings[col] = mapping
        X[col] = X[col].map(mapping)
        
    # Target encoding if classification
    y_mapping = None
    y_inverse_mapping = None
    if task_type == "classification" and y.dtype == "object":
        log_step("Encoding categorical target values...")
        y = y.fillna("Missing").astype(str)
        unique_targets = sorted(y.unique().tolist())
        y_mapping = {val: idx for idx, val in enumerate(unique_targets)}
        y_inverse_mapping = {idx: val for idx, val in enumerate(unique_targets)}
        y = y.map(y_mapping)
    elif task_type == "classification":
        # Check integer target mapping
        unique_targets = sorted(y.dropna().unique().tolist())
        y_inverse_mapping = {val: str(val) for val in unique_targets}
        
    # 3. Train-test split
    log_step(f"Splitting data into train ({100-int(test_size*100)}%) and test ({int(test_size*100)}%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "classification" else None
    )
    
    # 4. Feature Scaling
    scaler = None
    if scaling_method != "None" and len(num_feats) > 0:
        log_step(f"Applying feature scaling using {scaling_method}...")
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
            
        # Standardize train and test (fitting only on train!)
        X_train_num = scaler.fit_transform(X_train[num_feats])
        X_test_num = scaler.transform(X_test[num_feats])
        
        # Put back into dataframes
        X_train[num_feats] = X_train_num
        X_test[num_feats] = X_test_num
        
    # Show pipeline console animation
    console_ph = st.empty()
    for progress in range(len(selected_model_names) + 1):
        with console_ph.container():
            st.markdown("#### 🪵 Agent Logs")
            log_str = "\n".join(st.session_state.modeling_log)
            st.code(log_str, language="text")
        time.sleep(0.3)

    # 5. Fit selected models
    metrics_report = {}
    trained_models = {}
    
    for model_name in selected_model_names:
        log_step(f"Training {model_name}...")
        
        # Instantiate model
        if task_type == "classification":
            if model_name == "Random Forest":
                model = RandomForestClassifier(random_state=random_state, n_estimators=100, max_depth=6)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=random_state, n_estimators=100, max_depth=4)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(random_state=random_state, max_iter=1000)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(random_state=random_state, max_depth=6)
        else:
            if model_name == "Random Forest":
                model = RandomForestRegressor(random_state=random_state, n_estimators=100, max_depth=6)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(random_state=random_state, n_estimators=100, max_depth=4)
            elif model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Ridge Regression":
                model = Ridge(alpha=1.0)
                
        # Fit model
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        model_metrics = {}
        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            model_metrics["Accuracy"] = acc
            model_metrics["Precision"] = prec
            model_metrics["Recall"] = rec
            model_metrics["F1-Score"] = f1
            
            # ROC AUC if binary classification
            if len(np.unique(y_test)) == 2:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    model_metrics["ROC AUC"] = auc(fpr, tpr)
                    model_metrics["roc_curve"] = (fpr.tolist(), tpr.tolist())
                    
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            model_metrics["confusion_matrix"] = cm.tolist()
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            model_metrics["R2 Score"] = r2
            model_metrics["MAE"] = mae
            model_metrics["MSE"] = mse
            model_metrics["RMSE"] = rmse
            
        # Store predictions and actuals for plotting
        model_metrics["y_test"] = y_test.tolist()
        model_metrics["y_pred"] = y_pred.tolist()
        
        # Feature importance if available
        feat_importance = None
        if hasattr(model, "feature_importances_"):
            feat_importance = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            if task_type == "classification" and len(model.coef_) > 1:
                # Multi-class linear model coefficients average
                feat_importance = np.mean(np.abs(model.coef_), axis=0).tolist()
            else:
                feat_importance = np.abs(model.coef_).flatten().tolist()
                
        if feat_importance:
            model_metrics["feature_importances"] = dict(zip(features_list, feat_importance))
            
        metrics_report[model_name] = model_metrics
        log_step(f"{model_name} trained successfully.")

    log_step("All models evaluated. Pipeline complete!")
    
    # Store everything in session state
    st.session_state.model_metrics = metrics_report
    # Select the model with highest Accuracy (classification) or R2 (regression) as active
    best_model_name = None
    best_score = -99999
    metric_key = "Accuracy" if task_type == "classification" else "R2 Score"
    
    for m_name, m_data in metrics_report.items():
        score = m_data.get(metric_key, -9999)
        if score > best_score:
            best_score = score
            best_model_name = m_name
            
    st.session_state.trained_model = trained_models[best_model_name]
    st.session_state.trained_model_name = best_model_name
    st.session_state.model_features_list = features_list
    st.session_state.model_target_col = target_col
    st.session_state.model_scaler = scaler
    st.session_state.model_task_type = task_type
    st.session_state.model_encoded_categories = cat_mappings
    st.session_state.model_target_mapping = y_mapping
    st.session_state.model_target_inverse_mapping = y_inverse_mapping
    st.session_state.modeling_done = True
    
    # Redraw console logs once more
    with console_ph.container():
        st.markdown("#### 🪵 Agent Logs")
        log_str = "\n".join(st.session_state.modeling_log)
        st.code(log_str, language="text")

def display_results(task_type):
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
        
    comp_df = pd.DataFrame(comp_data)
    
    # Style dataframe
    st.dataframe(
        comp_df.style.highlight_max(
            subset=[col for col in comp_df.columns if col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC", "R2 Score"]],
            color="#064e3b"
        ).highlight_min(
            subset=[col for col in comp_df.columns if col in ["MAE", "MSE", "RMSE"]],
            color="#064e3b"
        ),
        use_container_width=True,
        hide_index=True
    )
    
    st.info(f"💡 The **{best_model_name}** model performed best and is selected for feature analysis and inference below.")
    
    # ── Model selection for detailed analysis ─────────────────────────────────
    st.markdown("### 🔍 Detailed Model Diagnostics")
    selected_model_name = st.selectbox("Select model for diagnostics", list(metrics.keys()))
    m_data = metrics[selected_model_name]
    
    col_metrics, col_plot = st.columns([1, 2])
    
    with col_metrics:
        st.markdown("#### 📊 Metrics Summary")
        for key, val in m_data.items():
            if isinstance(val, (int, float)):
                st.metric(key, f"{val:.4f}")
                
    with col_plot:
        if task_type == "classification":
            st.markdown("#### 🎯 Confusion Matrix")
            cm = np.array(m_data["confusion_matrix"])
            
            # Map target labels if mappings exist
            inverse_map = st.session_state.model_target_inverse_mapping
            if inverse_map:
                labels = [inverse_map.get(i, f"Class {i}") for i in range(len(cm))]
            else:
                labels = [f"Class {i}" for i in range(len(cm))]
                
            fig = px.imshow(
                cm,
                x=labels,
                y=labels,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                text_auto=True,
                color_continuous_scale="Purples",
                template="plotly_dark"
            )
            fig.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC curve if binary
            if "roc_curve" in m_data:
                st.markdown("#### 📈 ROC Curve")
                fpr, tpr = m_data["roc_curve"]
                roc_auc = m_data["ROC AUC"]
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.3f})', line=dict(color='#a78bfa', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='gray', dash='dash')))
                fig_roc.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=350,
                    margin=dict(t=10, b=10, l=10, r=10),
                    template="plotly_dark",
                    showlegend=True
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                
        else:
            st.markdown("#### 📉 Predicted vs Actual")
            y_test = np.array(m_data["y_test"])
            y_pred = np.array(m_data["y_pred"])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Data Points', marker=dict(color='#818cf8', opacity=0.7)))
            
            # Diagonal line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='y=x', line=dict(color='#f87171', dash='dash')))
            
            fig.update_layout(
                xaxis_title='Actual',
                yaxis_title='Predicted',
                height=350,
                margin=dict(t=10, b=10, l=10, r=10),
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📍 Residuals Plot")
            residuals = y_test - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals', marker=dict(color='#34d399', opacity=0.7)))
            fig_res.add_trace(go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0], mode='lines', name='Zero Line', line=dict(color='#64748b', dash='dash')))
            fig_res.update_layout(
                xaxis_title='Predicted (Fitted)',
                yaxis_title='Residuals',
                height=350,
                margin=dict(t=10, b=10, l=10, r=10),
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(fig_res, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    if "feature_importances" in m_data:
        st.markdown("---")
        st.markdown("### 📊 Feature Importance")
        feat_imp = m_data["feature_importances"]
        feat_imp_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        
        # Get top 15 features
        feat_imp_df = feat_imp_df.tail(15)
        
        fig_imp = px.bar(
            feat_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Predictor Importance in {selected_model_name}",
            color="Importance",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_imp.update_layout(height=400, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Predictions Playground ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎮 Predictions Playground")
    st.markdown("Simulate and run what-if scenarios! Provide custom values for features to get a real-time prediction.")
    
    # We will make predictions using the active trained model (which corresponds to st.session_state.trained_model)
    model = st.session_state.trained_model
    model_name = st.session_state.trained_model_name
    features = st.session_state.model_features_list
    scaler = st.session_state.model_scaler
    cat_mappings = st.session_state.model_encoded_categories
    inverse_target_map = st.session_state.model_target_inverse_mapping
    
    df_raw = get_df()
    
    # Split features into 2 columns for layout
    col_input1, col_input2 = st.columns(2)
    inputs = {}
    
    for idx, feat in enumerate(features):
        with col_input1 if idx % 2 == 0 else col_input2:
            # Check if feature is categorical in mapping
            if feat in cat_mappings:
                options = list(cat_mappings[feat].keys())
                inputs[feat] = st.selectbox(f"{feat}", options, key=f"play_{feat}")
            else:
                # Numeric feature: calculate min, max, median
                col_min = float(df_raw[feat].min())
                col_max = float(df_raw[feat].max())
                col_med = float(df_raw[feat].median())
                if np.isnan(col_min) or np.isnan(col_max):
                    col_min, col_max, col_med = 0.0, 100.0, 10.0
                    
                if col_min == col_max:
                    inputs[feat] = st.number_input(f"{feat} (Constant)", value=col_min, key=f"play_{feat}")
                else:
                    inputs[feat] = st.slider(f"{feat}", col_min, col_max, col_med, key=f"play_{feat}")

    if st.button("🔮 Run Prediction", use_container_width=True):
        # Format the inputs into a dataframe row
        input_row = pd.DataFrame([inputs])
        
        # Encode categorical variables
        for feat in cat_mappings:
            val = input_row[feat].iloc[0]
            val_encoded = cat_mappings[feat].get(val, cat_mappings[feat].get("Missing", 0))
            input_row[feat] = val_encoded
            
        # Scale numeric features if necessary
        if scaler:
            num_feats = df_raw[features].select_dtypes(include="number").columns.tolist()
            # Un-categorized features only are scaled
            num_feats = [f for f in num_feats if f not in cat_mappings]
            if num_feats:
                input_row[num_feats] = scaler.transform(input_row[num_feats])
                
        # Run prediction
        try:
            pred = model.predict(input_row)
            
            # Print output nicely
            st.markdown("<br>", unsafe_allow_html=True)
            if task_type == "classification":
                # Convert back if inverse mapped
                pred_label = pred[0]
                if inverse_target_map and pred_label in inverse_target_map:
                    pred_label = inverse_target_map[pred_label]
                    
                # Show probabilities if available
                prob_str = ""
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_row)[0]
                    # Map classes to indices
                    if inverse_target_map:
                        prob_details = [f"{inverse_target_map[i]}: {probs[i]*100:.1f}%" for i in range(len(probs))]
                    else:
                        prob_details = [f"Class {i}: {probs[i]*100:.1f}%" for i in range(len(probs))]
                    prob_str = "  \n**Class Probabilities:** " + " · ".join(prob_details)
                    
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #0f4c3a 0%, #0d2c24 100%); border: 1px solid #34d39944; border-radius: 10px; padding: 1.5rem; text-align: center;">
                    <div style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Prediction Output ({model_name})</div>
                    <div style="color:#34d399; font-size:2rem; font-weight:800;">{pred_label}</div>
                    <div style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">{prob_str}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                pred_val = pred[0]
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #1e3a8a 0%, #1e1b4b 100%); border: 1px solid #3b82f644; border-radius: 10px; padding: 1.5rem; text-align: center;">
                    <div style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Prediction Output ({model_name})</div>
                    <div style="color:#60a5fa; font-size:2.2rem; font-weight:800;">{pred_val:,.4f}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1a1936; border: 1px solid #2a2a4e; border-radius: 10px; padding: 1rem;">
        <p style="color:#94a3b8; margin:0; font-size:0.88rem;">
            ➡️ Next step: head to <strong style="color:#a78bfa;">💡 AI Insights</strong> for LLM-powered business and model interpretation.
        </p>
    </div>
    """, unsafe_allow_html=True)
