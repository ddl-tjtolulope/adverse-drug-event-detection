"""
Synthetic Adverse Drug Event Dataset Generator

Generates a realistic FAERS-inspired dataset for the ADE Detection Workshop.
Run this script locally if you do not have access to the Kaggle FAERS dataset.

Usage:
    python generate_dataset.py

Output:
    raw_ade_reports.csv  (~80,000 rows)
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 80_000

rng = np.random.default_rng(RANDOM_SEED)

DRUGS = {
    'warfarin':       'anticoagulant',
    'heparin':        'anticoagulant',
    'metformin':      'antidiabetic',
    'insulin':        'antidiabetic',
    'lisinopril':     'antihypertensive',
    'amlodipine':     'antihypertensive',
    'atorvastatin':   'statin',
    'simvastatin':    'statin',
    'ibuprofen':      'NSAID',
    'naproxen':       'NSAID',
    'amoxicillin':    'antibiotic',
    'ciprofloxacin':  'antibiotic',
    'sertraline':     'antidepressant',
    'fluoxetine':     'antidepressant',
    'valproate':      'antiepileptic',
    'carboplatin':    'chemotherapy',
    'paclitaxel':     'chemotherapy',
    'tacrolimus':     'immunosuppressant',
    'cyclosporine':   'immunosuppressant',
}

DRUG_NAMES  = list(DRUGS.keys())
DRUG_CLASSES = [DRUGS[d] for d in DRUG_NAMES]

ROUTES       = ['oral', 'IV', 'subcutaneous', 'intramuscular', 'topical']
INDICATIONS  = ['pain', 'hypertension', 'diabetes', 'infection', 'depression',
                 'cancer', 'transplant', 'hyperlipidemia', 'epilepsy', 'anticoagulation']
REPORTER_TYPES = ['physician', 'pharmacist', 'nurse', 'consumer']
REACTIONS    = ['cardiac', 'hepatic', 'allergic', 'neurological',
                'gastrointestinal', 'dermatological', 'hematological', 'renal']
RENAL_FUNCS  = ['normal', 'mild_impairment', 'moderate_impairment', 'severe_impairment']
HEPATIC_FUNCS = ['normal', 'mild_impairment', 'moderate_impairment']
SEXES        = ['Male', 'Female', 'Unknown']

# Drug-class-specific base serious rate (used to compute target)
DRUG_CLASS_SERIOUS_RATE = {
    'anticoagulant':    0.65,
    'chemotherapy':     0.80,
    'immunosuppressant':0.72,
    'antiepileptic':    0.55,
    'antidepressant':   0.38,
    'antihypertensive': 0.30,
    'antidiabetic':     0.35,
    'antibiotic':       0.25,
    'NSAID':            0.22,
    'statin':           0.18,
}

drug_idx      = rng.integers(0, len(DRUG_NAMES), N_ROWS)
drug_name_col = [DRUG_NAMES[i] for i in drug_idx]
drug_class_col = [DRUG_CLASSES[i] for i in drug_idx]

age        = np.clip(rng.normal(55, 16, N_ROWS), 18, 95).round(1)
sex        = rng.choice(SEXES, N_ROWS, p=[0.47, 0.48, 0.05])
weight_kg  = np.clip(rng.normal(78, 18, N_ROWS), 40, 160).round(1)

dose_mg    = np.clip(rng.lognormal(4.2, 0.9, N_ROWS), 1, 2000).round(1)
route      = rng.choice(ROUTES, N_ROWS, p=[0.60, 0.20, 0.08, 0.07, 0.05])
duration_days = np.clip(rng.exponential(30, N_ROWS), 1, 365).round().astype(int)
indication = rng.choice(INDICATIONS, N_ROWS)

concurrent_meds     = np.clip(rng.poisson(3.5, N_ROWS), 0, 12).astype(int)
reporter_type       = rng.choice(REPORTER_TYPES, N_ROWS, p=[0.35, 0.25, 0.20, 0.20])
time_to_onset_days  = np.clip(rng.exponential(12, N_ROWS), 0, 90).round().astype(int)
reaction_category   = rng.choice(REACTIONS, N_ROWS)
renal_function      = rng.choice(RENAL_FUNCS,  N_ROWS, p=[0.60, 0.20, 0.13, 0.07])
hepatic_function    = rng.choice(HEPATIC_FUNCS, N_ROWS, p=[0.70, 0.20, 0.10])
comorbidity_count   = np.clip(rng.poisson(2.5, N_ROWS), 0, 10).astype(int)
prior_ade           = rng.integers(0, 2, N_ROWS)

# Compute seriousness probability
base_prob = np.array([DRUG_CLASS_SERIOUS_RATE[dc] for dc in drug_class_col])
base_prob += (age > 65).astype(float) * 0.12
base_prob += (renal_function == 'severe_impairment').astype(float) * 0.15
base_prob += (renal_function == 'moderate_impairment').astype(float) * 0.08
base_prob += (hepatic_function == 'moderate_impairment').astype(float) * 0.08
base_prob += (concurrent_meds >= 6).astype(float) * 0.10
base_prob += (prior_ade == 1).astype(float) * 0.08
base_prob += (reaction_category.isin(['cardiac', 'hematological'])).astype(float) * 0.12  # type: ignore
base_prob += (reporter_type == 'physician').astype(float) * 0.05
base_prob += (time_to_onset_days <= 3).astype(float) * 0.06
base_prob += rng.normal(0, 0.05, N_ROWS)
base_prob = np.clip(base_prob, 0.02, 0.97)
serious = rng.binomial(1, base_prob).astype(int)

# Add ~2% missing values
def add_nulls(arr, pct=0.02):
    mask = rng.random(len(arr)) < pct
    result = arr.astype(object)
    result[mask] = np.nan
    return result

df = pd.DataFrame({
    'report_id':          range(1, N_ROWS + 1),
    'age':                add_nulls(age),
    'sex':                sex,
    'weight_kg':          add_nulls(weight_kg),
    'drug_name':          drug_name_col,
    'drug_class':         drug_class_col,
    'dose_mg':            add_nulls(dose_mg),
    'route':              route,
    'duration_days':      add_nulls(duration_days),
    'indication':         indication,
    'concurrent_meds':    concurrent_meds,
    'reporter_type':      reporter_type,
    'time_to_onset_days': time_to_onset_days,
    'reaction_category':  reaction_category,
    'renal_function':     renal_function,
    'hepatic_function':   hepatic_function,
    'comorbidity_count':  comorbidity_count,
    'prior_ade':          prior_ade,
    'Serious':            serious,
})

out_path = os.path.join(os.path.dirname(__file__), 'raw_ade_reports.csv')
df.to_csv(out_path, index=False)
print(f"Generated {len(df):,} rows → {out_path}")
print(f"Serious rate: {serious.mean():.1%}")
print(f"Null count per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
