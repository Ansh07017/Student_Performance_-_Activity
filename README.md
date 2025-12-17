# Student Lifestyle & Performance — Predictive Studio
Dynamic Link: https://studentperformanceandactivity.streamlit.app/ 

A fully interactive Streamlit app for exploratory data analysis and machine learning predictive modeling on student lifestyle datasets. Build regression models (CGPA prediction) and classification models (Stress level detection) with full model evaluation, interpretability, and export capabilities.

## Features

### 📊 **Three Main Pages**

1. **Overview & EDA**
   - Dataset metadata: rows, columns, null cells, data types
   - Interactive filters: by Stress Level, Study Category, Sleep Hours range, GPA range
   - Plotly visualizations: CGPA histogram, Sleep vs Stress box plot, Study vs CGPA scatter, correlation heatmap
   - Download filtered dataset as CSV

2. **Regression: Predict CGPA**
   - Build a complete pipeline: imputation → scaling → linear regression
   - 5-fold cross-validation with RMSE and R² metrics
   - Feature selection: choose which columns to include
   - Coefficient table (sorted by absolute effect)
   - Residuals plot and predicted vs actual scatter plot
   - SHAP explainability (if `shap` installed)
   - Live prediction panel: slide numeric features and select categorical values to get real-time CGPA predictions
   - Download trained model as `.pkl` file

3. **Classification: Predict Stress**
   - Train Random Forest classifier
   - Binary (High vs Not High) or multi-class (Low/Moderate/High) options
   - GridSearchCV hyperparameter tuning (n_estimators, max_depth)
   - Classification metrics: accuracy, precision, recall, F1
   - Confusion matrix heatmap
   - ROC curve and Precision-Recall curve (binary only)
   - Feature importance table
   - SHAP explainability for tree models
   - What-if / Live prediction: adjust features and see predicted stress class + probabilities
   - Download trained model as `.pkl` file
   - Export test set predictions as CSV

### 🎨 **UI & UX**

- **Dynamic gradient header**: deep navy/purple background with gradient styling
- **CSS background**: subtle blurred background image (ocean wave pattern via Unsplash)
- **Semi-transparent cards**: white background with 80% opacity for readability
- **Responsive layout**: uses `st.columns()`, `st.expander()`, and `st.tabs()`
- **Plotly interactive charts**: hover, zoom, pan, and download PNG
- **Mobile-friendly**: responsive design with fallbacks

### 🔧 **Data Preprocessing & Feature Engineering**

- **Missing value handling**: median imputation for numeric, most-frequent for categorical
- **Scaling**: StandardScaler for linear regression
- **Encoding**: One-hot encoding for categorical features
- **Derived features**:
  - `Sleep_Category`: Very Low / Low / Optimal / High
  - `Social_Bucket`: Low / Medium / High
  - `Activity_Level`: Low / Moderate / High (handles both `Physical_Activity_Hours_Per_Week` and `Physical_Activity_Hours_Per_Day`)
  - `Stress_Score`: numeric (0=Low, 1=Moderate, 2=High)

### 💾 **Model Persistence**

- Train models via Streamlit UI
- Download `.pkl` files with `joblib`
- Reload models later: `joblib.load("model.pkl")`

### 📥 **Data Input**

- Supports `student_lifestyle_dataset.csv` or `preprocessed_student_lifestyle_dataset.csv` in the same folder
- If no CSV found, generates a synthetic sample dataset (~200 rows)
- Automatically handles missing columns with graceful fallbacks

---

### Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone or download this project** to your local machine.

2. **Create a virtual environment** (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:

   ```powershell
   pip install -r requirements.txt
   ```

   This installs:
   - `streamlit`: main web framework
   - `pandas`, `numpy`: data processing
   - `scikit-learn`: ML models and utilities
   - `plotly`: interactive visualizations
   - `shap`: model interpretability (optional, graceful fallback if missing)
   - `joblib`: model serialization
   - `matplotlib`, `seaborn`, `eli5`: additional visualization/interpretation tools
   - `streamlit_lottie`: (optional) for header animations

---

## Running the App

### Quick Start

```powershell
streamlit run app.py
```

The app will open in your default browser at:
- **Local**: `http://localhost:8501`
- **Network**: `http://10.36.97.209:8501` (if accessible on your network)

### Custom Port

```powershell
streamlit run app.py --server.port 8503
```

---

## Usage Guide

### Overview & EDA Page
1. View dataset metadata and quick stats.
2. Use filters to slice the data by stress level, study category, sleep hours, or GPA.
3. Click **Apply** to update visualizations.
4. Download the filtered dataset via the link at the bottom.

### Regression Page
1. Select features from the multiselect box (defaults: Sleep, Study, Physical Activity, GPA).
2. The app trains a Linear Regression pipeline automatically.
3. Review 5-fold CV metrics and test set RMSE/R².
4. Scroll down to see:
   - Coefficient table (feature importance)
   - Residuals plot
   - Predicted vs Actual scatter
5. Use **Live prediction** sliders to input custom values and see predicted CGPA.
6. Download the trained model with the **Download trained regression model (.pkl)** button.

### Classification Page
1. Optionally toggle **Use multi-class** to predict all three stress levels instead of High vs Not High.
2. Select features for the classifier (defaults: Sleep, Study, GPA).
3. The app trains a RandomForest with GridSearchCV automatically.
4. Review:
   - Classification report (accuracy, precision, recall, F1)
   - Confusion matrix
   - ROC and PR curves (binary only)
   - Feature importance
5. Use **What-if / Live prediction** to input custom feature values and see:
   - Predicted stress class
   - Probability distribution
6. Download the model or export test set predictions as CSV.

---

## Architecture

```
app.py
├── load_data()                 # Load CSV or generate synthetic data
├── add_derived_features()      # Create Sleep_Category, Social_Bucket, Activity_Level, Stress_Score
├── get_download_link_df()      # Generate base64 CSV download links
├── css_background()            # Inject CSS for gradient header and background
│
└── Main Streamlit App:
    ├── Page 1: Overview & EDA
    │   ├── Dataset summary (shape, nulls, dtypes)
    │   ├── Interactive filter form
    │   ├── Plotly charts (histogram, box, scatter, heatmap)
    │   └── CSV download
    │
    ├── Page 2: Regression
    │   ├── Feature selection
    │   ├── Pipeline: impute → scale → LinearRegression
    │   ├── 5-fold CV evaluation
    │   ├── Coefficient extraction
    │   ├── Residuals and predicted vs actual plots
    │   ├── SHAP explainability (optional)
    │   ├── Live prediction form
    │   └── Model download (.pkl)
    │
    └── Page 3: Classification
        ├── Feature selection
        ├── Binary/multi-class target selection
        ├── GridSearchCV hyperparameter tuning
        ├── Classification metrics and confusion matrix
        ├── ROC/PR curves (binary)
        ├── Feature importance
        ├── SHAP explainability (optional)
        ├── What-if prediction form
        ├── Model download (.pkl)
        └── Predictions export (CSV)
```

---

## Example Datasets

The app assumes your dataset has these columns (or similar):

```
Student_ID, Sleep_Hours_Per_Day, Physical_Activity_Hours_Per_Week (or Per_Day),
Study_Hours_Per_Day, Social_Hours_Per_Day, GPA, CGPA, Study_Category, Stress_Level,
ExtraCurricular_Hours, Gender
```

If your dataset is missing any columns, the app will gracefully skip derived features or allow you to unselect unavailable features.

---

## Model Export & Reload

### Save a Model
Models are automatically downloadable as `.pkl` files from the respective pages.

### Load a Model Locally

```python
import joblib

# Load regression model
lr_model = joblib.load("linear_regression_cgpa.pkl")
X_new = pd.DataFrame([...])  # your new data
pred = lr_model.predict(X_new)

# Load classification model
rf_model = joblib.load("rf_stress_model.pkl")
pred = rf_model.predict(X_new)
pred_proba = rf_model.predict_proba(X_new)
```

---

## Troubleshooting

### KeyError: Column not found
- Ensure your CSV has the expected columns (Sleep_Hours_Per_Day, Study_Hours_Per_Day, GPA, etc.)
- The app gracefully handles missing columns; just deselect them in the feature multiselect.

### SHAP not available
- `shap` is optional. Install it with `pip install shap` if you want model explainability.
- The app will show an info message if `shap` is not installed.

### Models slow to train
- The app uses GridSearchCV with small parameter grids for speed.
- For larger datasets, reduce the sample size or increase grid search parameters in code.

### Port already in use
```powershell
streamlit run app.py --server.port 8503
```

---

## Advanced Configuration

### Customize Colors & Styling
Edit the `css_background()` function in `app.py` to change the gradient colors, background image, or card styles.

### Adjust Model Hyperparameters
In the Classification page, modify the `param_grid` dictionary for different tuning options:
```python
param_grid = {"rf__n_estimators":[50,100,200], "rf__max_depth":[5,10,15]}
```

### Change Background Image
Update the `BACKGROUND_IMAGE` URL in the app to any image you prefer:
```python
BACKGROUND_IMAGE = "https://your-image-url.com/photo.jpg"
```

---

## Dependencies

See `requirements.txt` for all packages. Main dependencies:
- **streamlit**: Web framework
- **pandas**, **numpy**: Data manipulation
- **scikit-learn**: ML algorithms (LinearRegression, RandomForestClassifier)
- **plotly**: Interactive visualizations
- **joblib**: Model serialization
- **shap** (optional): Model explainability

---

## License

This project is provided as-is for educational and research purposes.

---

## Support

For questions or issues:
1. Check the "How to use this studio" section in the Overview page.
2. Review the inline comments in `app.py`.
3. Ensure your dataset columns match the expected names.

---

## Future Enhancements

- [ ] Lottie animation in header
- [ ] Persistent model storage (save to `models/` folder)
- [ ] Advanced hyperparameter tuning UI
- [ ] Model comparison (ensemble methods, etc.)
- [ ] PDF/HTML report export
- [ ] Database integration for predictions logging

---

**Built with ❤️ for student lifestyle analysis and predictive modeling.**
