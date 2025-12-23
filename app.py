"""
Student Lifestyle & Performance — Predictive Studio

Multipage Streamlit app implementing:
- Overview & EDA
- Regression: Predict CGPA (Linear Regression)
- Classification: Predict High Stress (Random Forest)

Notes:
- Designed to run with `streamlit run app.py`.
- Assumes `student_lifestyle_dataset.csv` (or `preprocessed_student_lifestyle_dataset.csv`) exists
  in the same folder. If not found, the app will show a placeholder sample dataset.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import io
import base64
import joblib
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
try: 
  import shap
  SHAP_AVAILABLE = True 
except Exception: 
  SHAP_AVAILABLE = False 
# -----------------------------
# Assets (recommendations)
# -----------------------------
BACKGROUND_IMAGE = (
    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
)
LOTTIE_URL = "https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json"

# -----------------------------
# Utilities
# -----------------------------
def load_data(path_csv: str = "student_lifestyle_dataset.csv") -> pd.DataFrame:
    """Load dataset if present; otherwise return a small sample DataFrame."""
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
    elif os.path.exists("preprocessed_student_lifestyle_dataset.csv"):
        df = pd.read_csv("preprocessed_student_lifestyle_dataset.csv")
    else:
        # Small sample dataset with required columns
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "Student_ID": np.arange(1, n+1),
            "Sleep_Hours_Per_Day": np.round(rng.normal(7, 1.5, n).clip(3, 10), 1),
            "Physical_Activity_Hours_Per_Day": np.round(rng.normal(1, 0.5, n).clip(0, 3), 1),
            "Study_Hours_Per_Day": np.round(rng.normal(3, 1.5, n).clip(0, 12), 1),
            "Social_Hours_Per_Day": np.round(rng.normal(2, 1, n).clip(0, 8), 1),
            "GPA": np.round(rng.normal(3.2, 0.5, n).clip(1.0, 4.0), 2),
        })
        df["CGPA"] = df["GPA"] * 2.5
        df["Study_Category"] = pd.cut(df["Study_Hours_Per_Day"], bins=[-1,1.5,4,99], labels=["Learner","Studious","Very Studious"]).astype(str)
        df["Stress_Level"] = pd.cut( (10 - df["Sleep_Hours_Per_Day"]) + df["Study_Hours_Per_Day"]/2 + (4 - df["GPA"]), bins=3, labels=["Low","Moderate","High"]) .astype(str)
        df["ExtraCurricular_Hours"] = np.round(rng.normal(2, 1.5, n).clip(0, 10), 1)
        df["Gender"] = rng.choice(["Male","Female","Other"], size=n, p=[0.45,0.45,0.1])
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure CGPA exists (create from GPA if missing)
    if "CGPA" not in df.columns and "GPA" in df.columns:
        df["CGPA"] = df["GPA"] * 2.5
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Sleep category
    if "Sleep_Hours_Per_Day" in df.columns:
        df["Sleep_Category"] = pd.cut(df["Sleep_Hours_Per_Day"], bins=[-1,5,7,9,99], labels=["Very Low","Low","Optimal","High"]).astype(str)
    # Social bucket
    if "Social_Hours_Per_Day" in df.columns:
        df["Social_Bucket"] = pd.cut(df["Social_Hours_Per_Day"], bins=[-1,1,3,99], labels=["Low","Medium","High"]).astype(str)
    # Activity level (check both possible column names)
    activity_col = None
    if "Physical_Activity_Hours_Per_Week" in df.columns:
        activity_col = "Physical_Activity_Hours_Per_Week"
    elif "Physical_Activity_Hours_Per_Day" in df.columns:
        activity_col = "Physical_Activity_Hours_Per_Day"
    if activity_col:
        df["Activity_Level"] = pd.cut(df[activity_col], bins=[-1,1,3,99], labels=["Low","Moderate","High"]).astype(str)
    # Stress score numeric encoding (if Stress_Level exists)
    if "Stress_Level" in df.columns:
        df["Stress_Score"] = df["Stress_Level"].map({"Low":0, "Moderate":1, "High":2})
    return df

def get_download_link_df(df: pd.DataFrame, filename: str = "data.csv") -> tuple:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href, csv

# -----------------------------
# App main
# -----------------------------
def css_background(dark_mode: bool = False, image_url: str = BACKGROUND_IMAGE):
    """Inject CSS for light and dark themes. Uses doubled braces for f-string safety."""
    if not dark_mode:
        st.markdown(f"""
            <style>
            .stApp {{{{
                background-image: url('{image_url}');
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
            }}}}

            [data-testid="stAppViewContainer"] {{{{
                background-color: rgba(255, 255, 255, 0.995);
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 12px 40px rgba(11,27,43,0.08);
            }}}}

            h1, h2, h3, h4, h5, h6 {{{{
                color: #05204a !important;
                font-weight: 700;
            }}}}

            p, span, div, label {{{{
                color: #05204a !important;
                line-height: 1.45;
            }}}}

            [data-testid="stSidebar"] {{{{
                background-color: #ffffff;
                border-right: 1px solid #e6e9ef;
            }}}}

            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] * {{{{
                color: #05204a !important;
            }}}}

            [data-testid="metric-container"] {{{{
                background-color: #ffffff;
                padding: 12px;
                border-radius: 10px;
                border: 1px solid #eef2f7;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            }}}}

            .stButton > button {{{{
                background-color: #0077cc;
                color: #ffffff;
                border: none;
                padding: 8px 12px;
                border-radius: 8px;
                box-shadow: 0 6px 18px rgba(3,43,90,0.08);
            }}}}

            .stButton > button:hover {{{{
                background-color: #005fa3;
            }}}}

            .stForm {{{{
                background-color: #ffffff;
                padding: 18px;
                border-radius: 10px;
                border: 1px solid #e9eef6;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.03);
            }}}}

            input, select, textarea {{{{
                color: #05204a !important;
                background-color: #ffffff !important;
                border: 1px solid #d6dfea !important;
                border-radius: 6px !important;
            }}}}

            .stMultiSelect [data-baseweb="select"] {{{{
                background-color: #ffffff !important;
            }}}}

            .stSlider, .stSlider > div {{{{
                color: #05204a !important;
            }}}}

            .stCaption, .css-1v3fvcr p {{{{
                color: #31588a !important;
            }}}}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {{
                background-color: #0d1117;
                color: #c9d1d9;
            }}
            [data-testid="stAppViewContainer"] {{
                background-color: #0d1117;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #58a6ff !important;
            }}
            p, span, div, label {{
                color: #c9d1d9 !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: #161b22;
            }}
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] * {{
                color: #c9d1d9 !important;
            }}
            [data-testid="metric-container"] {{
                background-color: #161b22;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #30363d;
            }}
            .stButton > button {{
                background-color: #58a6ff;
                color: #0d1117;
                border: none;
            }}
            .stButton > button:hover {{
                background-color: #79c0ff;
            }}
            .stForm {{
                background-color: #161b22;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #30363d;
            }}
            input, select, textarea {{
                color: #c9d1d9 !important;
                background-color: #0d1117 !important;
                border: 1px solid #30363d !important;
            }}
            .stMultiSelect [data-baseweb="select"] {{
                background-color: #0d1117 !important;
            }}
            .stSlider {{
                color: #c9d1d9 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
st.set_page_config(page_title="Student Lifestyle & Performance — Predictive Studio", layout="wide")
css_background(dark_mode=False)  # Always use light theme (day mode)

# Load data with error handling
try:
    df = load_data()
    df = add_derived_features(df)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

st.sidebar.title("🎓 Student Lifestyle & Performance")
st.sidebar.markdown("Interactive studio for EDA and predictive modeling")
st.sidebar.divider()

page = st.sidebar.radio(
    "📊 Select Page:",
    ["📈 Overview & EDA", "📉 Regression: Predict CGPA", "🤖 Classification: Predict Stress"]
)

st.markdown(
    '<div class="header-gradient"><h1>🎓 Student Lifestyle & Performance — Predictive Studio</h1></div>',
    unsafe_allow_html=True
)

if page == "📈 Overview & EDA":
    st.subheader("Overview & Quick EDA")
    with st.expander("How to use this studio", expanded=True):
        st.write("""
        - Use the filters to explore the dataset.
        - Switch to Regression to build and evaluate a Linear Regression model for CGPA.
        - Switch to Classification to predict High stress using Random Forest.
        - Download models and predictions from the respective pages.
        """)

    # Dataset summary
    st.markdown("**Dataset snapshot**")
    st.dataframe(df.head())
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Null cells", int(df.isnull().sum().sum()))

    st.markdown("**Column types & missing**")
    meta = pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isnull().sum()})
    st.dataframe(meta)

    st.markdown("**Interactive Filters**")
    with st.form("eda_filters"):
        st.write("Filter dataset")
        sel_stress = st.multiselect("Stress_Level", options=df["Stress_Level"].unique().tolist(), default=df["Stress_Level"].unique().tolist())
        
        # Handle optional Study_Category column
        study_cat_opts = df["Study_Category"].unique().tolist() if "Study_Category" in df.columns else []
        sel_study = st.multiselect("Study_Category", options=study_cat_opts, default=study_cat_opts)
        
        sleep_min, sleep_max = st.slider("Sleep Hours Range", float(df["Sleep_Hours_Per_Day"].min()), float(df["Sleep_Hours_Per_Day"].max()), (float(df["Sleep_Hours_Per_Day"].min()), float(df["Sleep_Hours_Per_Day"].max())))
        gpa_min, gpa_max = st.slider("GPA Range", float(df["GPA"].min()), float(df["GPA"].max()), (float(df["GPA"].min()), float(df["GPA"].max())))
        submitted = st.form_submit_button("Apply")

    dff = df[(df["Stress_Level"].isin(sel_stress)) & (df["Sleep_Hours_Per_Day"].between(sleep_min, sleep_max)) & (df["GPA"].between(gpa_min, gpa_max))]
    
    # Apply Study_Category filter only if it exists
    if study_cat_opts and sel_study:
        dff = dff[dff["Study_Category"].isin(sel_study)]

    st.markdown("**Visualizations**")
    
    # Row 1: CGPA distribution and Study vs CGPA
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(dff, x="CGPA", nbins=20, title="CGPA distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        size_col = "ExtraCurricular_Hours" if "ExtraCurricular_Hours" in dff.columns else None
        fig3 = px.scatter(dff, x="Study_Hours_Per_Day", y="CGPA", color="Stress_Level", size=size_col, title="Study vs CGPA")
        st.plotly_chart(fig3, use_container_width=True)
    
    # Row 2: Sleep by Stress Level (full width, larger)
    fig2 = px.box(dff, x="Stress_Level", y="Sleep_Hours_Per_Day", title="Sleep by Stress Level", height=500)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Row 3: Key Features Correlation Matrix (full width, larger, coolwarm colors)
    corr_cols = ["Study_Hours_Per_Day", "Physical_Activity_Hours_Per_Day", "Social_Hours_Per_Day", "ExtraCurricular_Hours", "Sleep_Hours_Per_Day", "CGPA"]
    corr_cols = [c for c in corr_cols if c in dff.columns]  # Only include columns that exist
    if len(corr_cols) > 1:
        corr = dff[corr_cols].corr().round(2)  # Round to 2 decimal places
        fig4 = px.imshow(corr, text_auto=True, title="Key Features Correlation Matrix", color_continuous_scale="RdBu", height=600)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Not enough columns for correlation analysis")

    href, csv = get_download_link_df(dff)
    st.markdown(f"[Download filtered dataset]({href})")

elif page == "📉 Regression: Predict CGPA":
    st.subheader("Regression: Predict CGPA")
    st.markdown("Build a pipeline using **numeric features only** to predict CGPA with Linear Regression and 5-fold cross-validation.")

    # Get only numeric features (excluding CGPA and targets)
    numeric_features_list = df.select_dtypes(include=['number']).columns.tolist()
    numeric_features_list = [c for c in numeric_features_list if c not in ["CGPA", "Stress_Score", "Student_ID"]]
    
    if not numeric_features_list:
        st.warning("No numeric features found in dataset")
    else:
        # Feature selection - include Physical_Activity as default (important for student lifestyle)
        physical_activity_cols = [c for c in numeric_features_list if 'Physical_Activity' in c or 'Activity' in c]
        default_features = physical_activity_cols if physical_activity_cols else []
        # Add up to 3 more features if available
        other_features = [c for c in numeric_features_list if c not in default_features][:4]
        default_features = default_features + other_features
        chosen = st.multiselect("Select numeric features for CGPA prediction", options=numeric_features_list, default=default_features)

        if not chosen:
            st.warning("Please select at least one numeric feature")
        else:
            X = df[chosen].copy()
            y = df["CGPA"].values

            # Simple preprocessing: impute and scale
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            
            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            pipe = Pipeline([
                ("preproc", numeric_transformer),
                ("lr", LinearRegression())
            ])

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe.fit(X_train, y_train)

            # CV evaluation
            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")
            r2_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")

            y_pred = pipe.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Display metrics
            col1, col2, col3, col4= st.columns(4)
            col1.metric("CV RMSE (5-fold)", f"{(-cv_scores).mean():.3f}")
            col2.metric("CV R²", f"{r2_scores.mean():.3f}")
            col3.metric("Test RMSE", f"{rmse:.3f}")
            col4.metric("Test R²", f"{r2:.3f}")

            # Feature coefficients
            try:
                coefs = pipe.named_steps["lr"].coef_
                coef_df = pd.DataFrame({"Feature": chosen, "Coefficient": coefs})
                coef_df["Abs_Coef"] = coef_df["Coefficient"].abs()
                coef_df = coef_df.sort_values("Abs_Coef", ascending=False).drop("Abs_Coef", axis=1).reset_index(drop=True)
                
                with st.expander("📊 Feature Coefficients"):
                    st.dataframe(coef_df, use_container_width=True)
            except Exception as e:
                st.info("Could not extract coefficients")

            # Predicted vs Actual
            fig_pa = px.scatter(x=y_test, y=y_pred, labels={"x":"Actual CGPA","y":"Predicted CGPA"}, title="Predicted vs Actual CGPA")
            fig_pa.add_shape(type="line", x0=min(y_test), x1=max(y_test), y0=min(y_test), y1=max(y_test), line=dict(dash="dash", color="red"))
            st.plotly_chart(fig_pa, use_container_width=True)

            # SHAP if available
            if SHAP_AVAILABLE:
                with st.expander("🔍 SHAP Analysis (Optional)"):
                    try:
                        X_train_scaled = pipe.named_steps["preproc"].transform(X_train)
                        explainer = shap.Explainer(pipe.named_steps["lr"], X_train_scaled)
                        shap_vals = explainer(pipe.named_steps["preproc"].transform(X_test))
                        st.pyplot(shap.plots.beeswarm(shap_vals, show=False))
                    except Exception as e:
                        st.info("SHAP analysis not available for this configuration")

            # Live Prediction Panel
            st.markdown("---")
            st.subheader("🎯 Live CGPA Prediction")
            st.info("Adjust the sliders below to get a real-time CGPA prediction based on your input values.")
            
            # Create input columns with proper layout
            st.markdown("**Feature Values:**")
            pred_cols = st.columns(min(2, max(1, len(chosen))))
            pred_inputs = {}
            
            for idx, feature in enumerate(chosen):
                col_idx = idx % len(pred_cols)
                with pred_cols[col_idx]:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    med_val = float(X[feature].median())
                    
                    # Use a stable widget key so slider state persists across reruns
                    safe_feature_key = feature.replace(' ', '_').replace('/', '_')
                    pred_inputs[feature] = st.slider(
                        label=f"**{feature}**",
                        min_value=min_val,
                        max_value=max_val,
                        value=med_val,
                        step=0.01,
                        key=f"reg_slider_cgpa_{safe_feature_key}"
                    )
                    st.caption(f"Range: {min_val:.2f} - {max_val:.2f}")
            
            # Predict button
            if st.button("🎯 Predict CGPA", key="btn_predict_cgpa"):
                try:
                    X_input = pd.DataFrame([pred_inputs])
                    cgpa_pred = pipe.predict(X_input)[0]
                    st.success(f"✅ **Predicted CGPA: {cgpa_pred:.2f}**")
                    
                    # Classify student
                    if cgpa_pred >= 8.5:
                        category = "🏆 **Achiever**"
                    elif cgpa_pred >= 7.0:
                        category = "📚 **Good Student**"
                    else:
                        category = "📖 **Average Student**"
                    st.info(f"Student Classification: {category}")
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

            # Download model
            st.markdown("---")
            buf = io.BytesIO()
            joblib.dump(pipe, buf)
            buf.seek(0)
            st.download_button("⬇️ Download CGPA Prediction Model (.pkl)", data=buf, file_name="linear_regression_cgpa.pkl")

elif page == "🤖 Classification: Predict Stress":
    st.subheader("Classification: Predict Student Stress Level")
    st.markdown("Use **numeric features only** to predict Stress Level (Low/Moderate/High) with Random Forest Classifier.")

    # Get only numeric features (excluding Stress_Level and targets)
    numeric_features_list = df.select_dtypes(include=['number']).columns.tolist()
    numeric_features_list = [c for c in numeric_features_list if c not in ["Stress_Score", "Student_ID"]]
    
    if not numeric_features_list:
        st.warning("No numeric features found in dataset")
    else:
        # Feature selection - include Physical_Activity as default (important for student lifestyle)
        physical_activity_cols = [c for c in numeric_features_list if 'Physical_Activity' in c or 'Activity' in c]
        default_features = physical_activity_cols if physical_activity_cols else []
        # Add up to 3 more features if available
        other_features = [c for c in numeric_features_list if c not in default_features][:4]
        default_features = default_features + other_features
        features = st.multiselect("Select numeric features for Stress prediction", options=numeric_features_list, default=default_features)

        if not features:
            st.warning("Please select at least one numeric feature")
        else:
            X = df[features].copy()
            y = df["Stress_Level"].values

            # Simple preprocessing: impute and scale
            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            pipe_clf = Pipeline([
                ("preproc", numeric_transformer),
                ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42))
            ])

            # Train/test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            pipe_clf.fit(X_train, y_train)

            # Cross-validation scores for better estimate of model performance
            from sklearn.model_selection import cross_val_score
            cv_acc_scores = cross_val_score(pipe_clf, X, y, cv=5, scoring='accuracy')
            cv_f1_scores = cross_val_score(pipe_clf, X, y, cv=5, scoring='f1_weighted')

            # Predictions
            y_pred = pipe_clf.predict(X_test)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            col1.metric("CV Accuracy (5-fold)", f"{cv_acc_scores.mean():.3f}")
            col2.metric("CV F1 Score (5-fold)", f"{cv_f1_scores.mean():.3f}")
            col3.metric("Test Accuracy", f"{acc:.3f}")
            col4.metric("Test F1 Score", f"{f1:.3f}")

            # Classification report
            with st.expander("📊 Detailed Classification Report"):
                st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=pipe_clf.named_steps["rf"].classes_)
            fig_cm = px.imshow(cm, x=pipe_clf.named_steps["rf"].classes_, y=pipe_clf.named_steps["rf"].classes_, 
                              text_auto=True, title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm, use_container_width=True)

            # Feature Importance
            try:
                importances = pipe_clf.named_steps["rf"].feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)
                
                with st.expander("🔑 Feature Importance"):
                    fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")
                    st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.info("Could not extract feature importances")

            # SHAP if available
            if SHAP_AVAILABLE:
                with st.expander("🔍 SHAP Analysis (Optional)"):
                    try:
                        X_train_scaled = pipe_clf.named_steps["preproc"].transform(X_train)
                        explainer = shap.TreeExplainer(pipe_clf.named_steps["rf"])
                        shap_vals = explainer.shap_values(X_train_scaled)
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[0]
                        st.pyplot(shap.summary_plot(shap_vals, X_train_scaled, show=False))
                    except Exception as e:
                        st.info("SHAP analysis not available for this configuration")

            # Live Prediction Panel
            st.markdown("---")
            st.subheader("🎯 Live Stress Level Prediction")
            st.info("Adjust the sliders below to get a real-time stress prediction based on your input values.")
            
            pred_cols = st.columns(min(2, len(features)))
            pred_inputs_clf = {}
            
            for idx, feature in enumerate(features):
                with pred_cols[idx % len(pred_cols)]:
                    pred_inputs_clf[feature] = st.slider(
                        feature,
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        value=float(X[feature].median()),
                        step=0.1,
                        key=f"clf_slider_{feature}"
                    )
            
            # Predict button
            if st.button("🎯 Predict Stress Level", key="btn_predict_stress"):
                try:
                    X_input = pd.DataFrame([pred_inputs_clf])
                    stress_pred = pipe_clf.predict(X_input)[0]
                    stress_prob = pipe_clf.predict_proba(X_input)[0]
                    
                    st.success(f"✅ **Predicted Stress Level: {stress_pred}**")
                    
                    # Stress emoji mapping
                    stress_emoji = {
                        "Low": "😊",
                        "Moderate": "😐",
                        "High": "😰"
                    }
                    emoji = stress_emoji.get(stress_pred, "")
                    st.markdown(f"### {emoji} {stress_pred} Stress")
                    
                    # Probability distribution
                    prob_df = pd.DataFrame({
                        "Stress Level": pipe_clf.named_steps["rf"].classes_,
                        "Probability": stress_prob
                    }).sort_values("Probability", ascending=False)
                    
                    st.markdown("**Probability Distribution:**")
                    fig_prob = px.bar(prob_df, x="Stress Level", y="Probability", color="Stress Level",
                                     title="Stress Level Probabilities", color_discrete_map={
                                         "Low": "#2ecc71",
                                         "Moderate": "#f39c12",
                                         "High": "#e74c3c"
                                     })
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    st.dataframe(prob_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

            # Download model and predictions
            st.markdown("---")
            buf2 = io.BytesIO()
            joblib.dump(pipe_clf, buf2)
            buf2.seek(0)
            st.download_button("⬇️ Download Stress Prediction Model (.pkl)", data=buf2, file_name="random_forest_stress.pkl")
            
            # Export predictions
            if st.button("📥 Export Test Predictions as CSV"):
                preds_df = X_test.copy()
                preds_df["y_true"] = y_test
                preds_df["y_pred"] = y_pred
                csv_out = preds_df.to_csv(index=False)
                st.download_button("Download CSV", data=csv_out, file_name=f"stress_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

