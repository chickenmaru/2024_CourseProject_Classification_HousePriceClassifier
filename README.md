# 📘 Overview

This project, House Pricing by Classification, aims to predict house price ranges based on multiple property features using various classification algorithms.
The dataset (house_price.csv) includes features such as area, construction year, material quality, and neighborhood.
Through data preprocessing, feature engineering, and machine learning model comparison, the project identifies the most efficient and accurate model for predicting house price categories.

Developed as part of a Systems Analysis and Design course, the project was implemented in Python using libraries like pandas, scikit-learn, matplotlib, and seaborn.

# ⚙️ Logic Flow
## STEP 1 — Data Loading

Load house_price.csv dataset.

Inspect columns, missing values, and summary statistics.

## STEP 2 — Exploratory Data Analysis (EDA)

Visualize SalePrice distribution and correlations using histograms, scatterplots, and heatmaps.

Define two price range labels:

sp1: divided by quartiles (Q1, Q2, Q3).

sp2: divided into custom equal-width ranges using numpy.linspace().

## STEP 3 — Data Preprocessing

Handle missing values:

Drop columns with >50% missing data (e.g., PoolQC, Alley).

Fill numeric missing values by median or mode based on context.

Drop irrelevant columns like Utilities.

Remove Id column.

## STEP 4 — Feature Engineering

Apply one-hot encoding to categorical variables.

Perform feature selection using multiple methods:

VarianceThreshold to remove low-variance features.

SelectKBest and mutual_info_classif to select top correlated features.

RFE (Recursive Feature Elimination) with RandomForestClassifier.

SelectFromModel using LinearSVC to select important predictors.

Identify common strong predictors:
['MasVnrArea', 'TotalBsmtSF', 'OverallQual', 'GrLivArea', 'LotArea', '1stFlrSF'].

## STEP 5 — Machine Learning Models

Construct five feature sets (X1–X5) based on selected variables.

Normalize features using StandardScaler.

Split dataset (85% training, 15% testing).

Train and evaluate the following models:

Gaussian Naive Bayes

KNeighborsClassifier

Logistic Regression

Decision Tree

Random Forest

Extra Trees

MLP (Neural Network)

Stacking Classifier (ensemble method)

Use accuracy score and confusion matrix to evaluate performance.

## STEP 6 — Discussion & Conclusion

Compare label encoding methods (sp1 vs sp2).

Examine feature engineering performance and model interpretability.

Discuss computation complexity and performance trade-offs.

# 📊 Results

Most classifiers achieved over 96% accuracy, except GaussianNB, which struggled with high-dimensional feature spaces.

The best-performing combination:

Feature Set 5 (6 key features)

K-Nearest Neighbors (KNN) classifier

Advantages of final model:

High accuracy with low computational cost.

Interpretable through feature proximity visualization.

Efficient for small to medium datasets.

# 🧩 Key Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
