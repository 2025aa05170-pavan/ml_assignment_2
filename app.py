import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Classification Models Comparison App")

# Load metrics
metrics_df = pd.read_csv("model/metrics.csv")

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

st.sidebar.header("Options")
selected_model = st.sidebar.selectbox("Select Model", list(model_files.keys()))

uploaded_file = st.file_uploader("Upload Test CSV (features only)", type="csv")

# Display metrics
st.subheader("Evaluation Metrics")
st.dataframe(metrics_df[metrics_df["Model"] == selected_model])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    model = joblib.load(model_files[selected_model])

    preds = model.predict(data)

    st.subheader("Predictions")
    st.write(preds)

    if "diagnosis" in data.columns:
        y_true = data["diagnosis"]
        cm = confusion_matrix(y_true, preds)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        st.text(classification_report(y_true, preds))
