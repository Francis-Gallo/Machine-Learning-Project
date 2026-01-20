🚦 Road Accident Mortality Prediction (Machine Learning)

This project focuses on predicting the likelihood of mortality during traffic accidents using supervised machine learning techniques.
The goal is to identify key factors contributing to fatal outcomes and build robust, interpretable models despite severe class imbalance.

🎯 Problem Statement

Traffic accident datasets are typically:

Highly imbalanced (few fatal cases)

Heterogeneous (user, vehicle, road, and environmental factors)

Sensitive to model bias

This project addresses these challenges by applying data preprocessing, resampling strategies, and model evaluation techniques to build reliable mortality prediction models.

📊 Dataset Overview

Four official accident-related datasets were merged:

Accident characteristics (caract)

Location data (lieux)

User data (usagers)

Vehicle data (vehicules)

Final Dataset

309,341 rows

54 features

Target variable: mortality (binary)

Extensive preprocessing was performed to:

Handle missing values

Ensure data consistency

Encode categorical variables

🧠 Modeling Approach

Three supervised learning models were trained and evaluated:

🔹 Logistic Regression

Test Accuracy: 0.98

Served as a strong baseline

Performance limited by class imbalance

🔹 Random Forest

Test Accuracy: 0.94

Best overall trade-off between performance and robustness

Improved significantly using SMOTE and decision threshold tuning

🔹 Neural Network

Test Accuracy: 0.75

Required careful normalization and resampling

Highlighted sensitivity of neural models to data imbalance

⚙️ Key Techniques Used

SMOTE for handling class imbalance

Cross-validation for robust performance estimation

Threshold tuning to optimize recall for minority class

Feature engineering and preprocessing for structured data

📈 Evaluation Focus

Rather than relying solely on accuracy, the project emphasizes:

Model robustness on imbalanced data

Sensitivity to fatal cases

Generalization across unseen samples

🧪 What I Learned

Practical challenges of imbalanced classification

When simple models outperform complex ones

Importance of resampling and threshold tuning

Trade-offs between interpretability and model complexity

End-to-end ML workflow on large, real-world datasets

🔮 Possible Improvements

Feature importance and explainability (e.g., SHAP)

Cost-sensitive learning approaches

Temporal or geographic analysis of accident patterns

Deployment as a decision-support tool

🔗 Context

This project complements my work on:

Applied machine learning for decision support

Data-driven risk analysis

Interpretable AI systems
