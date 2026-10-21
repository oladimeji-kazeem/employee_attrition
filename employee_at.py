# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import pickle

# %% [code] Cell 2
df = pd.read_csv("employee(5).csv")

# %% [code] Cell 3
df.head()

# %% [code] Cell 4
df.shape

# %% [code] Cell 5
df.isnull().sum()

# %% [code] Cell 6
X = df.drop(columns=["employee_id", "attrition", "attrition_probability_simulated"])
y = df["attrition"]

# %% [code] Cell 7
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# %% [code] Cell 8
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

# %% [code] Cell 9
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    ))
])

# %% [code] Cell 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [code] Cell 11
model_pipeline.fit(X_train, y_train)

# %% [code] Cell 12
y_pred = model_pipeline.predict(X_test)

# %% [code] Cell 13
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# %% [code] Cell 14
print(classification_report(y_test, y_pred))

# %% [code] Cell 15
auc_score = roc_auc_score(y_test, y_pred_proba)

# %% [code] Cell 16
print(f"ROC AUC Score: {auc_score:.4f}")

# %% [code] Cell 17
# Save the model as a pickle in a file
joblib.dump(model_pipeline, 'employee_attrition.pkl')

# Load the model from the file
#knn_from_joblib = joblib.load('filename.pkl')
