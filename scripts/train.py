import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA = "../data/clean/diabetes.csv"
OUT  = "../data/models.json"

df = pd.read_csv(DATA)
FEATURES = ["age", "sex", "bmi", "waist", "sbp", "dbp", "glucose", "hba1c"]
X = df[FEATURES].values
y = df["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

imp = SimpleImputer(strategy="median")
X_train_i = imp.fit_transform(X_train)
X_test_i  = imp.transform(X_test)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train_i)
X_test_s  = sc.transform(X_test_i)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_s, y_train)

pred  = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)[:, 1]

metrics = {
    "accuracy":  round(accuracy_score(y_test, pred) * 100, 1),
    "precision": round(precision_score(y_test, pred) * 100, 1),
    "recall":    round(recall_score(y_test, pred) * 100, 1),
    "f1":        round(f1_score(y_test, pred) * 100, 1),
    "auc":       round(roc_auc_score(y_test, proba), 3),
    "n_train":   int(len(y_train)),
    "n_test":    int(len(y_test)),
    "n_total":   int(len(y)),
}
print("Held-out test metrics:", json.dumps(metrics, indent=2))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")
print(f"5-fold CV AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

payload = {
    "diabetes": {
        "features": FEATURES,
        "intercept": float(model.intercept_[0]),
        "coef": {k: float(v) for k, v in zip(FEATURES, model.coef_[0])},
        "scaler_mean":  {k: float(v) for k, v in zip(FEATURES, sc.mean_)},
        "scaler_scale": {k: float(v) for k, v in zip(FEATURES, sc.scale_)},
        "impute_median":{k: float(v) for k, v in zip(FEATURES, imp.statistics_)},
        "metrics": metrics,
        "cv_auc_mean": round(float(cv_auc.mean()), 3),
        "cv_auc_std":  round(float(cv_auc.std()), 3),
    }
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2)
print("Wrote", OUT)