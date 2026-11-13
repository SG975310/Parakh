import os, urllib.request

CYCLES = {"2015": "_I", "2017": "_J"}
FILES = ["HDL"]
OUT = "../data/raw"
UA = {"User-Agent": "Mozilla/5.0"}

for year, suffix in CYCLES.items():
    for fname in FILES:
        local = os.path.join(OUT, f"{fname}{suffix}.XPT")
        if os.path.exists(local) and os.path.getsize(local) > 50_000:
            print("skip", local)
            continue
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{fname}{suffix}.XPT"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if b"HEADER RECORD" not in data[:100]:
                print("  bad ", url, f"({len(data)} bytes)")
                continue
            with open(local, "wb") as f:
                f.write(data)
            print("  ok  ", url, f"({len(data)//1024} KB)")
        except Exception as e:
            print("  fail", url, e)