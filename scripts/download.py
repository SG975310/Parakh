import os, urllib.request

CYCLES = {"2015": "_I", "2017": "_J"}
FILES = ["DEMO", "GLU", "GHB", "BMX", "BPX", "DIQ"]
OUT = "../data/raw"
UA = {"User-Agent": "Mozilla/5.0"}

def urls(fname, suffix, year):
    yield f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{fname}{suffix}.XPT"
    yield f"https://wwwn.cdc.gov/Nchs/Nhanes/{year}-{int(year)+1}/{fname}{suffix}.XPT"

os.makedirs(OUT, exist_ok=True)

for year, suffix in CYCLES.items():
    for fname in FILES:
        local = os.path.join(OUT, f"{fname}{suffix}.XPT")
        if os.path.exists(local) and os.path.getsize(local) > 100_000:
            print("skip", local)
            continue
        if os.path.exists(local):
            os.remove(local)
        ok = False
        for url in urls(fname, suffix, year):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) < 100_000 or b"HEADER RECORD" not in data[:100]:
                    print("  bad ", url, f"({len(data)} bytes)")
                    continue
                with open(local, "wb") as f:
                    f.write(data)
                print("  ok  ", url, f"({len(data)//1024} KB)")
                ok = True
                break
            except Exception as e:
                print("  fail", url, e)
        if not ok:
            print("!! COULD NOT DOWNLOAD", fname + suffix)