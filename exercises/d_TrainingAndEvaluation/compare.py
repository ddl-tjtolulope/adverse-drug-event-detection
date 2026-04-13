# compare.py
import json
from pathlib import Path


def read_input(name: str) -> str:
    p = Path(f"/workflow/inputs/{name}")
    return p.read_text().strip() if p.exists() else name


ada_blob = json.loads(read_input("ada_results"))
gnb_blob = json.loads(read_input("gnb_results"))
xgb_blob = json.loads(read_input("xgb_results"))

consolidated = {"AdaBoost": ada_blob, "GaussianNB": gnb_blob, "XGBoost": xgb_blob}
print('Consolidated results received')

best_model, best_metric = '', 0.0
for name, blob in consolidated.items():
    if blob['roc_auc'] > best_metric:
        print(f'New best model: {name}  ROC-AUC={blob["roc_auc"]:.4f}')
        best_model  = name
        best_metric = blob['roc_auc']

print(f'Best model: {best_model}  ROC-AUC={best_metric:.4f}')

OUT_DIR  = Path("/workflow/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "consolidated"
OUT_FILE.write_text(f'model with highest AUC - {best_model}')
