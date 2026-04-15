"""
ADE Workshop — S3 Data Upload Script

Generates the synthetic adverse drug event dataset and uploads it to the
configured S3 bucket so that the Domino Data Source can serve it to participants.

Usage:
    python .setup/upload_to_s3.py \
        --bucket  <your-s3-bucket-name> \
        --region  us-east-2 \
        [--prefix ade-workshop/]   # optional key prefix inside the bucket

Requirements:
    pip install boto3
    AWS credentials must be configured (env vars, ~/.aws/credentials, or IAM role)
"""

import argparse
import io
import os
import sys

import boto3
import numpy as np
import pandas as pd

# ── Dataset generation (same logic as generate_dataset.py) ───────────────────

RANDOM_SEED = 42
N_ROWS      = 80_000

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

DRUG_NAMES   = list(DRUGS.keys())
DRUG_CLASSES = [DRUGS[d] for d in DRUG_NAMES]

ROUTES         = ['oral', 'IV', 'subcutaneous', 'intramuscular', 'topical']
INDICATIONS    = ['pain', 'hypertension', 'diabetes', 'infection', 'depression',
                  'cancer', 'transplant', 'hyperlipidemia', 'epilepsy', 'anticoagulation']
REPORTER_TYPES = ['physician', 'pharmacist', 'nurse', 'consumer']
REACTIONS      = ['cardiac', 'hepatic', 'allergic', 'neurological',
                  'gastrointestinal', 'dermatological', 'hematological', 'renal']
RENAL_FUNCS    = ['normal', 'mild_impairment', 'moderate_impairment', 'severe_impairment']
HEPATIC_FUNCS  = ['normal', 'mild_impairment', 'moderate_impairment']
SEXES          = ['Male', 'Female', 'Unknown']

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


def generate_dataset(n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    drug_idx       = rng.integers(0, len(DRUG_NAMES), n_rows)
    drug_name_col  = [DRUG_NAMES[i]  for i in drug_idx]
    drug_class_col = [DRUG_CLASSES[i] for i in drug_idx]

    age           = np.clip(rng.normal(55, 16, n_rows), 18, 95).round(1)
    sex           = rng.choice(SEXES, n_rows, p=[0.47, 0.48, 0.05])
    weight_kg     = np.clip(rng.normal(78, 18, n_rows), 40, 160).round(1)
    dose_mg       = np.clip(rng.lognormal(4.2, 0.9, n_rows), 1, 2000).round(1)
    route         = rng.choice(ROUTES, n_rows, p=[0.60, 0.20, 0.08, 0.07, 0.05])
    duration_days = np.clip(rng.exponential(30, n_rows), 1, 365).round().astype(int)
    indication    = rng.choice(INDICATIONS, n_rows)

    concurrent_meds    = np.clip(rng.poisson(3.5, n_rows), 0, 12).astype(int)
    reporter_type      = rng.choice(REPORTER_TYPES, n_rows, p=[0.35, 0.25, 0.20, 0.20])
    time_to_onset_days = np.clip(rng.exponential(12, n_rows), 0, 90).round().astype(int)
    reaction_category  = rng.choice(REACTIONS, n_rows)
    renal_function     = rng.choice(RENAL_FUNCS,   n_rows, p=[0.60, 0.20, 0.13, 0.07])
    hepatic_function   = rng.choice(HEPATIC_FUNCS, n_rows, p=[0.70, 0.20, 0.10])
    comorbidity_count  = np.clip(rng.poisson(2.5, n_rows), 0, 10).astype(int)
    prior_ade          = rng.integers(0, 2, n_rows)

    base_prob = np.array([DRUG_CLASS_SERIOUS_RATE[dc] for dc in drug_class_col])
    base_prob += (age > 65).astype(float) * 0.12
    base_prob += (renal_function == 'severe_impairment').astype(float) * 0.15
    base_prob += (renal_function == 'moderate_impairment').astype(float) * 0.08
    base_prob += (hepatic_function == 'moderate_impairment').astype(float) * 0.08
    base_prob += (concurrent_meds >= 6).astype(float) * 0.10
    base_prob += (prior_ade == 1).astype(float) * 0.08
    base_prob += np.isin(reaction_category, ['cardiac', 'hematological']).astype(float) * 0.12

    base_prob += (reporter_type == 'physician').astype(float) * 0.05
    base_prob += (time_to_onset_days <= 3).astype(float) * 0.06
    base_prob += rng.normal(0, 0.05, n_rows)
    base_prob  = np.clip(base_prob, 0.02, 0.97)
    serious    = rng.binomial(1, base_prob).astype(int)

    def add_nulls(arr, pct=0.02):
        mask   = rng.random(len(arr)) < pct
        result = arr.astype(object)
        result[mask] = np.nan
        return result

    return pd.DataFrame({
        'report_id':          range(1, n_rows + 1),
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


# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_dataframe_to_s3(df: pd.DataFrame, bucket: str, key: str, region: str) -> None:
    """Serialise a DataFrame to CSV in-memory and upload to S3."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    s3 = boto3.client('s3', region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    size_mb = buf.tell() / 1024 / 1024
    print(f"Uploaded s3://{bucket}/{key}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Generate ADE dataset and upload to S3")
    parser.add_argument('--bucket',  required=True,        help='S3 bucket name')
    parser.add_argument('--region',  default='us-east-2',  help='AWS region (default: us-east-2)')
    parser.add_argument('--prefix',  default='',           help='Optional key prefix inside the bucket')
    parser.add_argument('--n-rows',  type=int, default=N_ROWS, help=f'Number of rows to generate (default: {N_ROWS})')
    args = parser.parse_args()

    prefix = args.prefix.rstrip('/') + '/' if args.prefix else ''

    print(f"Generating synthetic ADE dataset ({args.n_rows:,} rows)...")
    df = generate_dataset(n_rows=args.n_rows)
    print(f"Generated {len(df):,} rows  |  Serious rate: {df['Serious'].mean():.1%}")
    print(f"Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

    raw_key = f"{prefix}raw_ade_reports.csv"
    upload_dataframe_to_s3(df, bucket=args.bucket, key=raw_key, region=args.region)

    print("\nDone. Configure the Domino Data Source to point to:")
    print(f"  Bucket : {args.bucket}")
    print(f"  Region : {args.region}")
    print(f"  File   : {raw_key}")
    print(f"\nData Source name to use in the workshop: adverse-drug-event-detection")


if __name__ == '__main__':
    main()
