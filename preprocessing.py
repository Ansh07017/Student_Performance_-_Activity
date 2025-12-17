import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as Standardscalar

df=pd.read_csv("student_lifestyle_dataset.csv")
#new comment
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['Stress_Level'].value_counts())
#create new column, CGPA=2.5*GPA
df['CGPA'] = df['GPA'] * 2.5
print(df['CGPA'].describe())
# add another categorical feature ofr CGPA,<6.5 average,<8.5 good and >8.5 achiever
def categorize_cgpa(cgpa):
    if cgpa < 6.5:
        return 'Average'
    elif cgpa < 8.5:
        return 'Good'
    else:
        return 'Achiever'
df['CGPA_Category'] = df['CGPA'].apply(categorize_cgpa)
print(df['CGPA_Category'].value_counts())
# Scaling
scaler = Standardscalar.StandardScaler()
numeric_features = ["Sleep_Hours_Per_Day","Physical_Activity_Hours_Per_Day","Social_Hours_Per_Day","Study_Hours_Per_Day"]
df[numeric_features] = scaler.fit_transform(df[numeric_features])
print(df.head())
# Save the preprocessed data
df.to_csv("preprocessed_student_lifestyle_dataset.csv", index=False)
