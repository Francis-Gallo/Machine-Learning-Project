Machine Learning Accident Analysis
Overview
This project predicts the likelihood of mortality during traffic accidents using machine learning models. The analysis combines accident, user, location, and vehicle data to identify patterns and predict outcomes.

Dataset
Four datasets (caract, lieux, usagers, vehicules) merged into a single dataset with 309,341 rows and 54 columns.
Cleaned and preprocessed to remove missing values and ensure data consistency.

Models and Performance

Logistic Regression:
Testing Accuracy: 0.98.
Faced challenges with class imbalance.

Random Forest:
Testing Accuracy: 0.94.
Best overall performance after using SMOTE and threshold tuning.

Neural Network:
Testing Accuracy: 0.75.
Improved with normalized and balanced data.

Key Features
Mortality prediction (mortality feature).
SMOTE applied to handle class imbalance.
Cross-validation and threshold tuning improved robustness
