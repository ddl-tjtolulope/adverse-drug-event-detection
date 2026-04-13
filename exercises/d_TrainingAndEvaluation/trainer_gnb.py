# trainer_gnb.py
from pathlib import Path
import json
from sklearn.naive_bayes import GaussianNB
from exercises.d_TrainingAndEvaluation.generic_trainer import train_ade

try:
    transformed_df_filename = Path("/workflow/inputs/transformed_filename").read_text().strip()
    print('using workflow input: transformed_filename', transformed_df_filename)
except FileNotFoundError as e:
    print('file not found error', e)
    transformed_df_filename = 'transformed_ade_reports.csv'

model_name = 'GaussianNB'
model_obj  = GaussianNB()

res = train_ade(model_obj, model_name, transformed_df_filename)

DROP  = {"threshold_scan", "curves"}
small = {k: v for k, v in res.items() if k not in DROP}

print(f"Training {model_name} completed successfully")
print(json.dumps({k: small.get(k) for k in ['roc_auc', 'f1_serious', 'accuracy', 'log_loss']}, indent=2))

out_path = Path("/workflow/outputs/results")
if out_path.parent.exists():
    out_path.write_text(json.dumps(small))
