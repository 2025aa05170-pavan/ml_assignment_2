import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Classification App", layout="wide")

st.title("Breast Cancer Classification - Model Comparison")


# Load Metrics
metrics_df = pd.read_csv("model/metrics.csv")

# Model mapping
model_files = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}

# Sidebar
st.sidebar.header("Model Selection")
selected_model_display = st.sidebar.selectbox(
    "Select Model",
    list(model_files.keys())
)

selected_model = model_files[selected_model_display]

# Show All Models Comparison
st.subheader("All Models Performance Comparison")
st.dataframe(metrics_df)

# Show Selected Model Metrics
st.subheader(f"{selected_model_display} - Detailed Metrics")
selected_metrics = metrics_df[metrics_df["Model"] == selected_model]
st.dataframe(selected_metrics)

# Confusion Matrix
st.subheader("Confusion Matrix")
cm_df = pd.read_csv(f"model/{selected_model}_cm.csv")
fig, ax = plt.subplots()
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
report_df = pd.read_csv(f"model/{selected_model}_report.csv")
st.dataframe(report_df)

# Prediction Section
st.subheader("Upload CSV for Prediction")
uploaded_file = st.file_uploader("Upload Test CSV (features only)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    model = joblib.load(f"model/{selected_model}.pkl")
    scaler = joblib.load("model/scaler.pkl")
    data_scaled = scaler.transform(data)
    preds = model.predict(data_scaled)
    st.subheader("Predictions")
    st.write(preds)
