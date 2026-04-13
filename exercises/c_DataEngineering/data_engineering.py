"""
Adverse Drug Event (ADE) Detection — Preprocessing Pipeline

This script performs data preprocessing for ADE seriousness classification:
1. Loads cleaned ADE report data from a CSV file
2. Creates derived features to enhance seriousness detection capabilities
3. Applies preprocessing transformations (scaling numerical features, encoding categorical features)
4. Generates an EDA (Exploratory Data Analysis) report
5. Saves the transformed features and logs everything to MLflow

The pipeline is designed to work within the Domino Data Lab platform and uses
MLflow for experiment tracking and model logging.
"""

import io, os, time
import pandas as pd
import numpy as np
import mlflow
from domino_data.data_sources import DataSourceClient
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature
from domino_short_id import domino_short_id


# Configure experiment name with a unique identifier to avoid conflicts
experiment_name = f"ADE Preprocessing {domino_short_id()}"

# Define filenames for input and output data
clean_filename    = 'clean_ade_reports.csv'       # Input: cleaned ADE report data
features_filename = 'transformed_ade_reports.csv' # Output: preprocessed features

# Get Domino environment paths (defaults provided for local development)
domino_working_dir   = os.environ.get("DOMINO_WORKING_DIR", ".")
domino_project_name  = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")
domino_project_owner = os.environ.get("DOMINO_PROJECT_OWNER",
                        os.environ.get("DOMINO_USER_NAME", "default-owner"))

domino_dataset_dir  = f"{domino_working_dir.replace('code', 'data')}/{domino_project_name}"
domino_artifact_dir = domino_working_dir.replace('code', 'artifacts')
clean_path          = f"{domino_dataset_dir}/{clean_filename}"


# ── Drug-class risk tier lookup ───────────────────────────────────────────────
DRUG_CLASS_RISK = {
    'chemotherapy':     'very_high',
    'immunosuppressant':'very_high',
    'anticoagulant':    'high',
    'antiepileptic':    'high',
    'antidepressant':   'medium',
    'antihypertensive': 'medium',
    'antidiabetic':     'medium',
    'antibiotic':       'low',
    'NSAID':            'low',
    'statin':           'low',
}

RENAL_SCORE  = {'normal': 0, 'mild_impairment': 1, 'moderate_impairment': 2, 'severe_impairment': 3}
HEPATIC_SCORE = {'normal': 0, 'mild_impairment': 1, 'moderate_impairment': 2}


def get_organ_impairment_score(row):
    """Combined organ impairment score (0–5). Higher = more impaired."""
    return RENAL_SCORE.get(row['renal_function'], 0) + HEPATIC_SCORE.get(row['hepatic_function'], 0)


def get_onset_speed(days):
    """Categorise time-to-onset into clinical buckets."""
    if days <= 3:
        return 'rapid'
    elif days <= 14:
        return 'moderate'
    else:
        return 'delayed'


def get_age_category(age):
    """Broad clinical age bands."""
    if age < 36:
        return 'young_adult'
    elif age < 65:
        return 'middle_aged'
    else:
        return 'elderly'


def get_reporter_credibility(reporter_type):
    """Proxy for signal quality based on reporter profession."""
    if reporter_type in ('physician', 'pharmacist'):
        return 'high'
    elif reporter_type == 'nurse':
        return 'medium'
    else:
        return 'low'


def get_polypharmacy_risk(concurrent_meds):
    """Polypharmacy risk tier based on number of concurrent medications."""
    if concurrent_meds <= 2:
        return 'low'
    elif concurrent_meds <= 5:
        return 'moderate'
    else:
        return 'high'


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features to enhance ADE seriousness detection.

    Args:
        df: Input dataframe with original ADE report features.

    Returns:
        Dataframe with additional derived features.

    Derived Features:
        - high_risk_age_flag:      1 if patient age > 65 (geriatric risk flag)
        - dose_weight_ratio:       dose_mg / weight_kg (relative exposure)
        - organ_impairment_score:  combined renal + hepatic impairment (0–5)
        - drug_class_risk_tier:    categorical risk tier for the drug class
        - reporter_credibility:    signal quality proxy (high/medium/low)
        - onset_speed:             time-to-onset category (rapid/moderate/delayed)
        - age_category:            clinical age band
        - polypharmacy_risk:       risk tier based on concurrent medications
    """
    df = df.copy()

    # Geriatric flag — age > 65 is a well-established ADE risk factor
    df['high_risk_age_flag'] = (df['age'] > 65).astype(int)

    # Relative drug exposure (dose normalised by body weight)
    # Adding small epsilon to avoid division by zero
    df['dose_weight_ratio'] = df['dose_mg'] / (df['weight_kg'] + 1e-6)

    # Organ impairment score: compromised kidneys/liver affect drug clearance
    df['organ_impairment_score'] = df.apply(get_organ_impairment_score, axis=1)

    # Drug class risk tier — encodes domain knowledge about drug safety profiles
    df['drug_class_risk_tier'] = df['drug_class'].map(DRUG_CLASS_RISK).fillna('unknown')

    # Reporter credibility — physician/pharmacist reports are more reliable signals
    df['reporter_credibility'] = df['reporter_type'].apply(get_reporter_credibility)

    # Onset speed — rapid onset often indicates more serious reactions
    df['onset_speed'] = df['time_to_onset_days'].apply(get_onset_speed)

    # Clinical age band
    df['age_category'] = df['age'].apply(get_age_category)

    # Polypharmacy risk — many concurrent medications increases interaction risk
    df['polypharmacy_risk'] = df['concurrent_meds'].apply(get_polypharmacy_risk)

    return df


if __name__ == "__main__":

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="ADE Preprocessing Pipeline") as run:

        # Step 1: Load cleaned ADE reports
        print(f"Loading clean dataset from {clean_path}")
        clean_df = pd.read_csv(clean_path, index_col=0)
        print(f"Loaded {len(clean_df):,} rows")
        print(clean_df.columns.tolist())

        # Step 2: Engineer derived features
        full_df = add_derived_features(clean_df)

        # Step 3: Separate target variable
        labels_df   = full_df['Serious']
        features_df = full_df.drop(columns=['Serious', 'report_id', 'drug_name'], errors='ignore')

        # Identify feature types
        numeric_features     = features_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = features_df.select_dtypes(include=[object, 'category']).columns.tolist()
        print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

        # Step 4: Build preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        pipeline = Pipeline([("preproc", preprocessor)])

        # Fit and transform
        start_time = time.time()
        transformed_array = pipeline.fit_transform(features_df)
        fit_time = time.time() - start_time
        print(f"Transformed features in {fit_time:.2f} seconds.")

        # Step 5: Rebuild DataFrame with named columns
        feature_names = pipeline.named_steps['preproc'].get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_array, columns=feature_names, index=features_df.index)
        transformed_df['Serious'] = labels_df

        # Step 6: Save transformed features
        features_path = f"{domino_dataset_dir}/{features_filename}"
        transformed_df.to_csv(features_path, index=False)
        print(f"Saved to {features_path}")

        # Step 7: Generate EDA report
        from ydata_profiling import ProfileReport
        profile = ProfileReport(
            clean_df,
            title="Adverse Drug Event Detection — EDA Report",
            explorative=True,
            minimal=True
        )
        eda_path = f"{domino_artifact_dir}/preprocessing_report.html"
        profile.to_file(eda_path)

        # Step 8: Log everything to MLflow
        mlflow.log_artifact(clean_path, artifact_path="data")
        mlflow.log_artifact(eda_path,   artifact_path="eda")
        mlflow.log_param("num_rows_loaded",      len(features_df))
        mlflow.log_param("num_cat_features",     len(categorical_features))
        mlflow.log_param("num_num_features",     len(numeric_features))
        mlflow.log_metric("fit_time",            fit_time)

        # Step 9: Log preprocessing pipeline as an MLflow model
        pipeline.predict = pipeline.transform

        X_sample = features_df.iloc[:20].copy()
        for col in numeric_features:
            if np.issubdtype(X_sample[col].dtype, np.integer):
                X_sample[col] = X_sample[col].astype("float64")

        y_sample  = pipeline.transform(X_sample)
        signature = infer_signature(X_sample, y_sample)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="ade_preprocessing_pipeline",
            signature=signature
        )

        mlflow.set_tag("pipeline", "ade_preprocessing")
