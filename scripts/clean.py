import os
import pandas as pd
import numpy as np

RAW = "../data/raw"
OUT = "../data/clean"
os.makedirs(OUT, exist_ok=True)

MISSING = [7, 9, 77, 99, 777, 999, 7777, 9999]

def load(name):
    return pd.read_sas(os.path.join(RAW, name), format="xport")

frames = []
for s in ["_I", "_J"]:
    demo = load(f"DEMO{s}.XPT")
    glu  = load(f"GLU{s}.XPT")
    ghb  = load(f"GHB{s}.XPT")
    bmx  = load(f"BMX{s}.XPT")
    bpx  = load(f"BPX{s}.XPT")
    diq  = load(f"DIQ{s}.XPT")

    d = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
    d = d.merge(glu[["SEQN", "LBXGLU"]], on="SEQN", how="left")
    d = d.merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
    d = d.merge(bmx[["SEQN", "BMXBMI", "BMXWAIST"]], on="SEQN", how="left")
    d = d.merge(bpx[["SEQN", "BPXSY1", "BPXDI1"]], on="SEQN", how="left")
    d = d.merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={
    "RIDAGEYR": "age", "RIAGENDR": "sex",
    "LBXGLU": "glucose", "LBXGH": "hba1c",
    "BMXBMI": "bmi", "BMXWAIST": "waist",
    "BPXSY1": "sbp", "BPXDI1": "dbp",
    "DIQ010": "diq_diabetes",
})

for col in ["glucose", "hba1c", "bmi", "waist", "sbp", "dbp", "diq_diabetes"]:
    df.loc[df[col].isin(MISSING), col] = np.nan

df = df[df["age"] >= 20].copy()

has_self = df["diq_diabetes"].isin([1, 2])
has_lab  = df["glucose"].notna() | df["hba1c"].notna()
df = df[has_self | has_lab].copy()

df["diabetes"] = (
    (df["diq_diabetes"] == 1)
    | (df["glucose"] >= 126)
    | (df["hba1c"] >= 6.5)
).astype(int)

keep = ["SEQN", "age", "sex", "bmi", "waist", "sbp", "dbp",
        "glucose", "hba1c", "diabetes"]
df = df[keep]
df.to_csv(os.path.join(OUT, "diabetes.csv"), index=False)

print("total rows:", len(df))
print("positive (diabetes):", int(df["diabetes"].sum()))
print("negative:", int((df["diabetes"] == 0).sum()))
print()
print("missing per column:")
print(df.isna().sum())