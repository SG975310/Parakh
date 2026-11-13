import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

CLEAN = "../data/clean"
OUT_MODELS = "../data/models.json"
OUT_CARD   = "../data/model_card.json"

MODELS_DEF = {
    "diabetes": {
        "csv": "diabetes.csv", "label": "diabetes",
        "features": ["age","sex","bmi","waist","sbp","dbp","glucose","hba1c"],
        "limitations": ["Glucose and HbA1c are both in the label definition and the feature set: the model partly reads the lab report that defines the label.",
                        "No cross-cycle validation available (single combined cohort)."],
    },
    "heart": {
        "csv": "heart.csv", "label": "heart",
        "features": ["age","sex","bmi","sbp","dbp","chol","hdl","trig","hba1c","crp",
                     "smoker","diabetes","active_days","bp_diagnosed","family_hx"],
        "limitations": ["Label is self-reported heart disease from a survey — noisy (people forget or were never diagnosed).",
                        "High-band precision is inherently lower than other models because the label itself is imprecise."],
    },
    "kidney": {
        "csv": "kidney.csv", "label": "kidney",
        "features": ["age","sex","bun","acr","sbp","uric_acid","albumin",
                     "potassium","calcium","phosphorus","chloride","hba1c","diabetes","htn"],
        "limitations": ["Label is eGFR < 60 (KDIGO stage 3a+), which is itself a derived quantity. Creatinine was deliberately excluded from the feature set to avoid a direct label leak."],
    },
    "fattyliver": {
        "csv": "fattyliver.csv", "label": "fatty_liver",
        "features": ["age","sex","alt","ast","alp","ggt","bili","albumin",
                     "hdl","glucose","tg_hdl_ratio","diabetes"],
        "limitations": ["Label is Fatty Liver Index (FLI) ≥ 60, a validated proxy score, not an ultrasound or FibroScan diagnosis. TG, BMI, and waist are excluded from features because they define FLI — that would be a label leak."],
    },
}

def threshold_at_recall(y, p, target):
    best = 0.5
    for t in np.linspace(0.02, 0.98, 97):
        if recall_score(y, (p >= t).astype(int), zero_division=0) >= target:
            best = t
    return round(float(best), 3)

def score_at(y, p, t):
    pred = (p >= t).astype(int)
    return {
        "accuracy":  round(accuracy_score(y, pred)*100, 1),
        "precision": round(precision_score(y, pred, zero_division=0)*100, 1),
        "recall":    round(recall_score(y, pred, zero_division=0)*100, 1),
        "f1":        round(f1_score(y, pred, zero_division=0)*100, 1),
    }

def bootstrap_ci(y, p, n_boot=500):
    rng = np.random.default_rng(42)
    scores = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        try:
            scores.append(roc_auc_score(y[idx], p[idx]))
        except Exception:
            pass
    if not scores: return (0.5, 0.5)
    return (round(float(np.percentile(scores, 2.5)), 3),
            round(float(np.percentile(scores, 97.5)), 3))

def sigmoid_fn(x, a, b):
    return 1.0 / (1.0 + np.exp(-(a*x + b)))

def net_benefit(y, p, thresholds):
    n = len(y); out = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = ((pred==1) & (y==1)).sum()
        fp = ((pred==1) & (y==0)).sum()
        nb = tp/n - fp/n * (t/(1-t))
        out.append(round(float(nb), 4))
    return out

payload, cards = {}, {}

for name, cfg in MODELS_DEF.items():
    print(f"\n=== {name} ===")
    df = pd.read_csv(os.path.join(CLEAN, cfg["csv"]))
    F, y_name = cfg["features"], cfg["label"]
    X, y = df[F].values, df[y_name].values
    has_cycle = "cycle" in df.columns

    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_te_i = imp.transform(X_te)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_i)
    X_te_s = sc.transform(X_te_i)

    base = LogisticRegression(max_iter=3000)
    base.fit(X_tr_s, y_tr)

    cal = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    cal.fit(X_tr_s, y_tr)

    # Fit a single Platt curve mapping base logit -> calibrated prob (approx)
    try:
        logits_tr = base.decision_function(X_tr_s)
        cal_tr = cal.predict_proba(X_tr_s)[:, 1]
        popt, _ = curve_fit(sigmoid_fn, logits_tr, cal_tr, p0=[1, 0], maxfev=20000)
        a_platt, b_platt = float(popt[0]), float(popt[1])
    except Exception:
        a_platt, b_platt = 1.0, 0.0

    logits_te = base.decision_function(X_te_s)
    proba = sigmoid_fn(logits_te, a_platt, b_platt)

    auc = roc_auc_score(y_te, proba)
    auc_ci = bootstrap_ci(y_te, proba)
    print(f"AUC (calibrated): {auc:.3f}  [95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f}]")

    t_mod = threshold_at_recall(y_te, proba, 0.90)
    t_high = threshold_at_recall(y_te, proba, 0.60)
    m_mod = score_at(y_te, proba, t_mod)
    m_high = score_at(y_te, proba, t_high)
    print(f"  moderate (90% recall): t={t_mod}  {m_mod}")
    print(f"  high (60% recall):     t={t_high}  {m_high}")

    # Cross-cycle validation (train on I, test on J)
    xcycle = None
    if has_cycle:
        cyc = df["cycle"].values
        cyc_tr, cyc_te = cyc[idx_tr], cyc[idx_te]
        mI, mJ = cyc_tr == "I", cyc_te == "J"
        if mI.sum() > 200 and mJ.sum() > 200:
            try:
                cm = LogisticRegression(max_iter=3000)
                cm.fit(X_tr_s[mI], y_tr[mI])
                cp = cm.predict_proba(X_te_s[mJ])[:, 1]
                if len(np.unique(y_te[mJ])) > 1:
                    cauc = roc_auc_score(y_te[mJ], cp)
                    xcycle = {"train": "I", "test": "J",
                              "n_train": int(mI.sum()), "n_test": int(mJ.sum()),
                              "auc": round(float(cauc), 3)}
                    print(f"  cross-cycle I→J: AUC {cauc:.3f} (n_test={mJ.sum()})")
            except Exception as e:
                print("  cross-cycle skipped:", e)

    # Subgroup AUC
    subgroups = {}
    df_te = df.iloc[idx_te]
    if "sex" in F:
        for lab, mask in [("female", df_te["sex"]==2), ("male", df_te["sex"]==1)]:
            mask = mask.values
            if mask.sum() > 50 and len(np.unique(y_te[mask])) > 1:
                subgroups[f"sex_{lab}"] = {"n": int(mask.sum()),
                                            "auc": round(float(roc_auc_score(y_te[mask], proba[mask])), 3)}
    if "age" in F:
        age = df_te["age"].values
        for lab, mask in [("age_under_50", age<50), ("age_50plus", age>=50)]:
            if mask.sum() > 50 and len(np.unique(y_te[mask])) > 1:
                subgroups[lab] = {"n": int(mask.sum()),
                                  "auc": round(float(roc_auc_score(y_te[mask], proba[mask])), 3)}
    print(f"  subgroups: {subgroups}")

    # Decision curve analysis
    dca_t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    prev = y_te.sum() / len(y_te)
    nb_model = net_benefit(y_te, proba, dca_t)
    nb_all = [round(float(prev - (1-prev)*(t/(1-t))), 4) for t in dca_t]
    dca = {"thresholds": dca_t, "model": nb_model, "treat_all": nb_all,
           "treat_none": [0.0]*len(dca_t)}

    # CV AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(base, X_tr_s, y_tr, cv=cv, scoring="roc_auc")
    print(f"  cv auc: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

    payload[name] = {
        "features": F,
        "intercept": float(base.intercept_[0]),
        "coef": {k: float(v) for k, v in zip(F, base.coef_[0])},
        "scaler_mean": {k: float(v) for k, v in zip(F, sc.mean_)},
        "scaler_scale": {k: float(v) for k, v in zip(F, sc.scale_)},
        "impute_median": {k: float(v) for k, v in zip(F, imp.statistics_)},
        "platt_a": a_platt, "platt_b": b_platt,
        "threshold_moderate": t_mod, "threshold_high": t_high,
        "auc": round(float(auc), 3),
        "auc_ci": list(auc_ci),
        "metrics_moderate": m_mod, "metrics_high": m_high,
        "cross_cycle": xcycle, "subgroups": subgroups, "dca": dca,
        "n_total": int(len(y)), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "prevalence": round(float(y.mean())*100, 1),
    }

    cards[name] = {
        "name": name, "features": F, "source_csv": cfg["csv"],
        "n_total": int(len(y)), "prevalence_pct": round(float(y.mean())*100, 1),
        "auc": round(float(auc), 3), "auc_ci_95": list(auc_ci),
        "cv_auc_mean": round(float(cv_auc.mean()), 3),
        "cv_auc_std":  round(float(cv_auc.std()), 3),
        "operating_points": {
            "moderate_90pct_recall": {"threshold": t_mod, **m_mod},
            "high_60pct_recall":     {"threshold": t_high, **m_high},
        },
        "cross_cycle_validation": xcycle,
        "subgroup_auc": subgroups,
        "decision_curve": dca,
        "known_limitations": cfg["limitations"],
    }

with open(OUT_MODELS, "w") as f: json.dump(payload, f, indent=2)
with open(OUT_CARD, "w") as f: json.dump(cards, f, indent=2)
print(f"\nWrote {OUT_MODELS}")
print(f"Wrote {OUT_CARD}")