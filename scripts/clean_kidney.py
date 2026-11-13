import os, pandas as pd, numpy as np

RAW = "../data/raw"; OUT = "../data/clean"
MISSING = [7, 9, 77, 99, 777, 999, 7777, 9999]

def load(n): return pd.read_sas(os.path.join(RAW, n), format="xport")

def egfr_2021(scr, age, sex):
    female = (sex == 2)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    ratio = scr / k
    egfr = 142 * np.minimum(ratio,1)**a * np.maximum(ratio,1)**-1.200 * 0.9938**age
    return egfr * np.where(female, 1.012, 1.0)

frames = []
for s in ["_I", "_J"]:
    demo   = load(f"DEMO{s}.XPT")
    biopro = load(f"BIOPRO{s}.XPT")
    albcr  = load(f"ALB_CR{s}.XPT")
    bpx    = load(f"BPX{s}.XPT")
    diq    = load(f"DIQ{s}.XPT")
    ghb    = load(f"GHB{s}.XPT")

    d = demo[["SEQN","RIDAGEYR","RIAGENDR"]].copy()
    d = d.merge(biopro[["SEQN","LBXSCR","LBXSBU","LBXSUA","LBXSAL","LBXSKSI","LBXSCA","LBXSPH","LBXSCLSI"]], on="SEQN", how="left")
    d = d.merge(albcr[["SEQN","URDACT"]], on="SEQN", how="left")
    d = d.merge(bpx[["SEQN","BPXSY1"]], on="SEQN", how="left")
    d = d.merge(diq[["SEQN","DIQ010"]], on="SEQN", how="left")
    d = d.merge(ghb[["SEQN","LBXGH"]], on="SEQN", how="left")
    d["cycle"] = "I" if s == "_I" else "J"
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={
    "RIDAGEYR":"age","RIAGENDR":"sex",
    "LBXSCR":"creatinine","LBXSBU":"bun","LBXSUA":"uric_acid",
    "LBXSAL":"albumin","LBXSKSI":"potassium","LBXSCA":"calcium",
    "LBXSPH":"phosphorus","LBXSCLSI":"chloride",
    "URDACT":"acr","BPXSY1":"sbp","DIQ010":"diq_diabetes","LBXGH":"hba1c",
})

for col in ["creatinine","bun","uric_acid","albumin","potassium","calcium",
            "phosphorus","chloride","acr","sbp","diq_diabetes","hba1c"]:
    df.loc[df[col].isin(MISSING), col] = np.nan

df = df[df["age"] >= 20].copy()
df = df[df["creatinine"].notna()].copy()

df["egfr"] = egfr_2021(df["creatinine"].values, df["age"].values, df["sex"].values)
df["kidney"] = (df["egfr"] < 60).astype(int)
df["diabetes"] = (df["diq_diabetes"]==1).astype(int)
df["htn"] = (df["sbp"] >= 140).astype(int)

keep = ["SEQN","age","sex","bun","acr","sbp","uric_acid","albumin",
        "potassium","calcium","phosphorus","chloride","hba1c",
        "diabetes","htn","egfr","kidney","cycle"]
df[keep].to_csv(os.path.join(OUT,"kidney.csv"), index=False)

print("kidney rows:", len(df))
print("positive:", int(df["kidney"].sum()))
print("by cycle:", df.groupby("cycle").size().to_dict())
print(df[keep].isna().sum())