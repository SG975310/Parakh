import os, pandas as pd, numpy as np

RAW = "../data/raw"; OUT = "../data/clean"
MISSING = [7, 9, 77, 99, 777, 999, 7777, 9999]

def load(n): return pd.read_sas(os.path.join(RAW, n), format="xport")

frames = []
for s in ["_I", "_J"]:
    demo  = load(f"DEMO{s}.XPT")
    bmx   = load(f"BMX{s}.XPT")
    bpx   = load(f"BPX{s}.XPT")
    tchol = load(f"TCHOL{s}.XPT")
    hdl   = load(f"HDL{s}.XPT")
    trig  = load(f"TRIGLY{s}.XPT")
    ghb   = load(f"GHB{s}.XPT")
    hscrp = load(f"HSCRP{s}.XPT")
    smq   = load(f"SMQ{s}.XPT")
    mcq   = load(f"MCQ{s}.XPT")
    diq   = load(f"DIQ{s}.XPT")
    paq   = load(f"PAQ{s}.XPT")
    bpq   = load(f"BPQ{s}.XPT")

    d = demo[["SEQN","RIDAGEYR","RIAGENDR"]].copy()
    d = d.merge(bmx[["SEQN","BMXBMI"]], on="SEQN", how="left")
    d = d.merge(bpx[["SEQN","BPXSY1","BPXDI1"]], on="SEQN", how="left")
    d = d.merge(tchol[["SEQN","LBXTC"]], on="SEQN", how="left")
    d = d.merge(hdl[["SEQN","LBDHDD"]], on="SEQN", how="left")
    d = d.merge(trig[["SEQN","LBXTR"]], on="SEQN", how="left")
    d = d.merge(ghb[["SEQN","LBXGH"]], on="SEQN", how="left")
    d = d.merge(hscrp[["SEQN","LBXHSCRP"]], on="SEQN", how="left")
    d = d.merge(smq[["SEQN","SMQ020"]], on="SEQN", how="left")
    d = d.merge(mcq[["SEQN","MCQ160B","MCQ160C","MCQ160E","MCQ300A"]], on="SEQN", how="left")
    d = d.merge(diq[["SEQN","DIQ010"]], on="SEQN", how="left")
    d = d.merge(paq[["SEQN","PAD615","PAD630"]], on="SEQN", how="left")
    d = d.merge(bpq[["SEQN","BPQ020"]], on="SEQN", how="left")
    d["cycle"] = "I" if s == "_I" else "J"
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={
    "RIDAGEYR":"age","RIAGENDR":"sex",
    "BMXBMI":"bmi","BPXSY1":"sbp","BPXDI1":"dbp",
    "LBXTC":"chol","LBDHDD":"hdl","LBXTR":"trig","LBXGH":"hba1c",
    "LBXHSCRP":"crp",
    "SMQ020":"smoked",
    "MCQ160B":"chd","MCQ160C":"mi","MCQ160E":"angina",
    "DIQ010":"diq_diabetes",
})

for col in ["bmi","sbp","dbp","chol","hdl","trig","hba1c","crp","smoked",
            "chd","mi","angina","diq_diabetes","MCQ300A"]:
    df.loc[df[col].isin(MISSING), col] = np.nan

for col in ["PAD615","PAD630"]:
    df.loc[df[col].isin([77, 99]), col] = np.nan
df["active_days"] = df[["PAD615","PAD630"]].max(axis=1)

df.loc[df["BPQ020"].isin([7, 9]), "BPQ020"] = np.nan
df["bp_diagnosed"] = df["BPQ020"].map({1.0: 1, 2.0: 0})
df["family_hx"] = df["MCQ300A"].map({1.0: 1, 2.0: 0})

df = df[df["age"] >= 20].copy()
answered = df[["chd","mi","angina"]].isin([1,2]).any(axis=1)
df = df[answered].copy()

df["heart"] = ((df["chd"]==1) | (df["mi"]==1) | (df["angina"]==1)).astype(int)
df["smoker"] = (df["smoked"]==1).astype(int)
df["diabetes"] = (df["diq_diabetes"]==1).astype(int)

keep = ["SEQN","age","sex","bmi","sbp","dbp","chol","hdl","trig","hba1c","crp",
        "smoker","diabetes","active_days","bp_diagnosed","family_hx","heart","cycle"]
df[keep].to_csv(os.path.join(OUT,"heart.csv"), index=False)

print("heart rows:", len(df))
print("positive:", int(df["heart"].sum()))
print("by cycle:", df.groupby("cycle").size().to_dict())
print(df[keep].isna().sum())