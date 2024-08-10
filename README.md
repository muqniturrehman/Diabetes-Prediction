# Diabetes Prediction Project

This project aims to predict the likelihood of diabetes in individuals using various machine learning models, specifically Logistic Regression and Random Forest Classifier. The dataset used for this project is sourced from Kaggle and contains several medical predictor variables as well as an outcome variable indicating whether the individual has diabetes.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest Classifier](#random-forest-classifier)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

## Project Overview

This project implements a machine learning pipeline to predict diabetes based on a given dataset. The process involves data preprocessing, exploratory data analysis (EDA), model training, and evaluation using different metrics like accuracy, precision, recall, F1-score, and ROC-AUC score.

## Dataset

The dataset used in this project is the "Kaggle Diabetes Dataset" which includes the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1) indicating whether the patient has diabetes

## Libraries Used

The project utilizes the following Python libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
```
## Data Preprocessing
- **Normalization**: Applied Min-Max Scaling to normalize the features to the range [0, 1].
```python
# Data Normalization using Min-Max Scaling
min_max_scaler = MinMaxScaler()
df1 = min_max_scaler.fit_transform(df[['Pregnancies', 'Glucose',
                                       'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                                       'DiabetesPedigreeFunction', 'Age']])
df1 = pd.DataFrame(df1, columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

```
- **Train-Test Split**: The dataset was split into a training set (80%) and a test set (20%).
```python
# Split Features (X) and Target Variable (y)
X = df1 # Exclude 'Outcome' column
y = df['Outcome'] # Target variable is 'Outcome'

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Exploratory Data Analysis (EDA)
The dataset was explored through various visualizations:
- **Histogram**: To understand the distribution of numeric features.
```python
df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].hist(bins=10, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Features")
plt.show()
```

- **Pairplot**: To visualize the relationship between different features and the outcome.
```python
sns.pairplot(df, hue='Outcome')
plt.title("Pairplot of Features with Outcome")
plt.show()

```
## Modeling
# Logistic Regression
-**Training**:  A Logistic Regression model was trained on the normalized training data.
```python
# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

```
- **Evaluation**: The model's performance was evaluated using accuracy, classification report, and confusion matrix.
```python
# Predict on the test data
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# Print classification report
print(classification_report(y_test, y_pred))
# Print confusion matrix
print(confusion_matrix(y_test, y_pred))
# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_auc}')

```
# Random Forest Classifier
-**Training**: A Random Forest Classifier was also trained for comparison with Logistic Regression.
```python
# Random Forest Classifier Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model on the training data
rf_classifier.fit(X_train, y_train)

```
- **Evaluation**:  Similar metrics (accuracy, classification report, confusion matrix) were used for evaluation.
```python
# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

```
## Evaluation
The models were evaluated based on the following metrics:
- **Accuracy**: The percentage of correct predictions out of all predictions made.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1-Score**: The weighted average of Precision and Recall.
- **ROC-AUC Score**: A measure of how well the model distinguishes between classes.
## Conclusion
This project demonstrated the effectiveness of Logistic Regression and Random Forest Classifiers in predicting diabetes. Random Forest generally provided better results in terms of accuracy and other metrics.


