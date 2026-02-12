# ML Assignment 2 – Classification Models & Streamlit Deployment

#a)  Problem Statement
The objective of this assignment is to implement multiple machine learning 
classification models on a real-world dataset, evaluate their performance using 
standard evaluation metrics, and deploy an interactive Streamlit web application 
to demonstrate the trained models.

This project covers the complete end-to-end machine learning workflow including:
- Data preprocessing
- Model training
- Model evaluation
- UI development using Streamlit
- Deployment on Streamlit Community Cloud

---

#b)  Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source:** Kaggle  

The dataset contains diagnostic features computed from digitized images of breast
mass. The goal is to classify tumors as:

- 0 → Benign  
- 1 → Malignant  

**Number of Instances:** 569  
**Number of Features:** 30 numerical features  
**Type:** Binary Classification  

This dataset satisfies assignment requirements:
- Minimum 500 instances
- Minimum 12 features 

---

## Machine Learning Models Implemented

The following 6 models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## Evaluation Metrics Used

For each model, the following metrics were calculated:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
--------------|----------|-----------|-----------|--------|----------|-----|
| Logistic Regression |0.965 |0.9962 |0.98 |0.9245 |0.9515 |0.9251 |
| Decision Tree |0.901 |0.9173 |0.9388 |0.8679 |0.902 |0.8493 |
| KNN |0.958 |0.986 |0.9736 |0.9057 |0.9412 |0.9103 |
| Naive Bayes |0.9441 |0.9925 |0.9592 |0.8868 |0.9216 |0.8798 |
| Random Forest |0.972 |0.9972 |1 |0.9245 |0.9608 |0.9408 |
| XGBoost |0.972 |0.9937 |1 |0.9245 |0.9608 |0.9408 |



---

# Observations

- Logistic Regression performed well due to relatively separable features.
- Decision Tree captured nonlinear patterns but may slightly overfit.
- KNN performance depends on choice of K and scaling.
- Naive Bayes provided stable but slightly lower performance.
- Random Forest improved performance using ensemble learning.
- XGBoost achieved strong performance due to boosting technique.

---

# Streamlit Application Features

- CSV Test Data Upload
- Model Selection Dropdown
- Display of Evaluation Metrics
- Confusion Matrix Visualization
- Classification Report Display

---

#Deployment

Streamlit App Link: (https://mazrhvybviki8dlnvwxmwj.streamlit.app/ and 
http://localhost:8501/)

GitHub Repository Link: (https://github.com/2025aa05170-pavan/ml_assignment_2.git)

