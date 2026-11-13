import os, pandas as pd, numpy as np

RAW = "../data/raw"; OUT = "../data/clean"
MISSING = [7, 9, 77, 99, 777, 999, 7777, 9999]

def load(n): return pd.read_sas(os.path.join(RAW, n), format="xport")

probe = load("BIOPRO_I.XPT")
GGT = next((c for c in ["LBXSGTSI","LBXSGT"] if c in probe.columns), None)
if GGT is None: print("no GGT"); raise SystemExit(1)

frames = []
for s in ["_I", "_J"]:
    demo   = load(f"DEMO{s}.XPT")
    bmx    = load(f"BMX{s}.XPT")
    biopro = load(f"BIOPRO{s}.XPT")
    trigly = load(f"TRIGLY{s}.XPT")
    diq    = load(f"DIQ{s}.XPT")
    hdl    = load(f"HDL{s}.XPT")
    glu    = load(f"GLU{s}.XPT")

    d = demo[["SEQN","RIDAGEYR","RIAGENDR"]].copy()
    d = d.merge(bmx[["SEQN","BMXBMI","BMXWAIST"]], on="SEQN", how="left")
    d = d.merge(biopro[["SEQN","LBXSASSI","LBXSATSI","LBXSAPSI",GGT,"LBXSTB","LBXSAL"]], on="SEQN", how="left")
    d = d.merge(trigly[["SEQN","LBXTR"]], on="SEQN", how="left")
    d = d.merge(diq[["SEQN","DIQ010"]], on="SEQN", how="left")
    d = d.merge(hdl[["SEQN","LBDHDD"]], on="SEQN", how="left")
    d = d.merge(glu[["SEQN","LBXGLU"]], on="SEQN", how="left")
    if GGT != "LBXSGT":
        d = d.rename(columns={GGT: "LBXSGT"})
    d["cycle"] = "I" if s == "_I" else "J"
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={
    "RIDAGEYR":"age","RIAGENDR":"sex",
    "BMXBMI":"bmi","BMXWAIST":"waist",
    "LBXSASSI":"alt","LBXSATSI":"ast",
    "LBXSAPSI":"alp","LBXSGT":"ggt",
    "LBXSTB":"bili","LBXSAL":"albumin",
    "LBXTR":"tg","DIQ010":"diq_diabetes",
    "LBDHDD":"hdl","LBXGLU":"glucose",
})

for col in ["bmi","waist","alt","ast","alp","ggt","bili","albumin","tg",
            "diq_diabetes","hdl","glucose"]:
    df.loc[df[col].isin(MISSING), col] = np.nan

df = df[df["age"] >= 20].copy()
mask = df[["tg","ggt","waist","bmi"]].notna().all(axis=1) & (df["tg"]>0) & (df["ggt"]>0)
df = df[mask].copy()

# TG/HDL ratio — metabolic syndrome marker, not part of FLI formula
df["tg_hdl_ratio"] = df["tg"] / df["hdl"].replace(0, np.nan)

def compute_fli(tg, ggt, waist, bmi):
    L = 0.953*np.log(tg) + 0.139*bmi + 0.718*np.log(ggt) + 0.053*waist - 15.745
    return 100 * np.exp(L) / (1 + np.exp(L))

df["fli"] = compute_fli(df["tg"].values, df["ggt"].values, df["waist"].values, df["bmi"].values)
df["fatty_liver"] = (df["fli"] >= 60).astype(int)
df["diabetes"] = (df["diq_diabetes"]==1).astype(int)

keep = ["SEQN","age","sex","alt","ast","alp","ggt","bili","albumin",
        "hdl","glucose","tg_hdl_ratio","diabetes","fli","fatty_liver","cycle"]
df[keep].to_csv(os.path.join(OUT,"fattyliver.csv"), index=False)

print("fattyliver rows:", len(df))
print("positive:", int(df["fatty_liver"].sum()))
print("by cycle:", df.groupby("cycle").size().to_dict())
print(df[keep].isna().sum())