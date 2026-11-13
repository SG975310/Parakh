import json
with open("../data/models.json") as f:
    models = json.load(f)
out = "// AUTO-GENERATED from data/models.json. Regenerate: python scripts/export_models.py\n\n"
out += "const MODELS = " + json.dumps(models, indent=2) + ";\n"
with open("../models.js", "w") as f:
    f.write(out)
print("Wrote ../models.js")
for name, m in models.items():
    print(f"  {name}: AUC {m['auc']}, {len(m['features'])} features")