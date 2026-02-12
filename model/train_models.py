import pandas as pd

import numpy as np

import joblib

from sklearn.impute import SimpleImputer


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (

    accuracy_score, roc_auc_score, precision_score,

    recall_score, f1_score, matthews_corrcoef

)



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Load dataset

df = pd.read_csv("data.csv")



# Drop unnecessary column if exists

if "id" in df.columns:

    df.drop("id", axis=1, inplace=True)



# Encode target

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})



X = df.drop("diagnosis", axis=1)

y = df["diagnosis"]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)



# Train-test split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=42, stratify=y

)



# Scaling


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)




models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(),

    "KNN": KNeighborsClassifier(),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(n_estimators=100),

    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")

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



    # Save model

    joblib.dump(model, f"model/{name.replace(' ', '_').lower()}.pkl")



# Save metrics

metrics_df = pd.DataFrame(results)

metrics_df.to_csv("model/metrics.csv", index=False)



print("Training complete. Models & metrics saved.")



