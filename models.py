# train_linear_regression.py(CGPA) and then random forest classifier(Stress),which then add to app.py,so we can run streamlit app
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
df = pd.read_csv("preprocessed_student_lifestyle_dataset.csv")
#then linear regression to check
print(df.head())
X = df.drop("Stress_Level", axis=1)
y = df["Stress_Level"]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Preprocessing pipelines
numeric_features = ["Sleep_Hours_Per_Day", "Physical_Activity_Hours_Per_Day", "Social_Hours_Per_Day", "Study_Hours_Per_Day"]
categorical_features = ["CGPA_Category"]
#Train Linear Regression for CGPA prediction
lr = LinearRegression()
lr.fit(X_train[numeric_features], df.loc[X_train.index, 'CGPA'])
# Predict CGPA on test set
cgpa_pred = lr.predict(X_test[numeric_features])
print("Linear Regression CGPA Predictions:")
print(cgpa_pred)
# One-hot encoding for categorical features
X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
# Align columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
# Evaluate model
y_pred = rf_clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Feature importance
importances = rf_clf.feature_importances_
feature_names = X_train.columns
feat_importances = pd.Series(importances, index=feature_names)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importances")
plt.show()
