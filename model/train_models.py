import pandas as pd
import numpy as np
import joblib
import os

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


os.makedirs("model", exist_ok=True)

df = pd.read_csv("data.csv")

if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")


models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    results.append(metrics)

    #Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(f"model/{name}_cm.csv", index=False)

    # Save Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"model/{name}_report.csv")

    # Save model
    joblib.dump(model, f"model/{name}.pkl")


# Save metrics comparison
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("model/metrics.csv", index=False)

print("Training complete. Models, confusion matrices, reports & metrics saved.")
