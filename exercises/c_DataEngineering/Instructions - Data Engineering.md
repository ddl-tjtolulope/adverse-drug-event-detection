# Exercise 3: Data Engineering

## Objective
Run an automated data preprocessing pipeline as a Domino Job. Apply pharmacovigilance-specific feature engineering, scale numerical features, encode categorical features, and produce a versioned dataset ready for model training.

---

## Steps

### 1. Review the Pipeline Script
Open `exercises/c_DataEngineering/data_engineering.py` and familiarise yourself with the derived features:

| Derived Feature | Description |
|---|---|
| `high_risk_age_flag` | 1 if patient age > 65 (geriatric pharmacovigilance risk) |
| `dose_weight_ratio` | Dose (mg) normalised by body weight — relative exposure metric |
| `organ_impairment_score` | Combined renal + hepatic impairment score (0–5) |
| `drug_class_risk_tier` | Categorical risk tier for drug class (low/medium/high/very_high) |
| `reporter_credibility` | Signal quality proxy based on reporter profession |
| `onset_speed` | Time-to-onset category (rapid ≤3 days / moderate / delayed) |
| `age_category` | Clinical age band (young_adult / middle_aged / elderly) |
| `polypharmacy_risk` | Risk tier based on concurrent medication count |

### 2. Run as a Domino Job
1. From your project, navigate to **Jobs → Run**
2. Set the command to:
   ```
   python exercises/c_DataEngineering/data_engineering.py
   ```
3. Select a hardware tier (Small is sufficient)
4. Click **Run**

### 3. Monitor the Job
1. Watch the job logs — you should see:
   - Dataset loaded successfully
   - Transformed features count
   - Time taken for preprocessing
   - Artifacts saved
2. Navigate to **Experiments** to view the MLflow run created by this job

### 4. Inspect the Outputs
- `transformed_ade_reports.csv` — saved to your Domino Dataset
- `preprocessing_report.html` — EDA report in **Artifacts**
- The preprocessing pipeline is logged as an MLflow model for deployment

---

## Key Concepts Demonstrated
- **Domino Jobs** — reproducible, scheduled batch execution
- **MLflow Integration** — automatic experiment tracking built into Domino
- **Feature Engineering for Pharmacovigilance** — domain-driven derived features
- **Preprocessing Pipelines as Models** — the scaler/encoder is versioned and deployable

---

## Relevant Documentation
- [Create and run Jobs](https://docs.dominodatalab.com/en/latest/user_guide/af97b7/create-and-run-jobs/)
