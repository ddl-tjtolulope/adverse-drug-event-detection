import os
import sys
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exercises.c_DataEngineering.data_engineering import add_derived_features

# Import configuration (fallback to environment variables if config file not available)
try:
    from app_config import (
        FEATURE_SCALING_ENDPOINT, FEATURE_SCALING_AUTH,
        MODEL_CONFIG
    )
    feature_scaling_endpoint = FEATURE_SCALING_ENDPOINT
    feature_scaling_auth     = FEATURE_SCALING_AUTH
    model_scaling_dict       = MODEL_CONFIG
    print("Loaded configuration from app_config.py")
except ImportError:
    print("app_config.py not found, using environment variables")
    feature_scaling_endpoint = os.environ.get('feature_scaling_endpoint', 'ENDPOINT_NOT_CONFIGURED')
    feature_scaling_auth     = os.environ.get('feature_scaling_auth',     'AUTH_NOT_CONFIGURED')

    model_scaling_dict = {
        'XG Boost': {
            'endpoint': os.environ.get('xgboost_endpoint',   'ENDPOINT_NOT_CONFIGURED'),
            'auth':     os.environ.get('xgboost_auth',       'AUTH_NOT_CONFIGURED'),
        },
        'ADA Boost': {
            'endpoint': os.environ.get('adaboost_endpoint',  'ENDPOINT_NOT_CONFIGURED'),
            'auth':     os.environ.get('adaboost_auth',      'AUTH_NOT_CONFIGURED'),
        },
        'GaussianNB': {
            'endpoint': os.environ.get('gaussiannb_endpoint','ENDPOINT_NOT_CONFIGURED'),
            'auth':     os.environ.get('gaussiannb_auth',    'AUTH_NOT_CONFIGURED'),
        }
    }

# ── Classifier input schema (must match preprocessing pipeline output) ────────
CLASSIFIER_SCHEMA = [
    'num__age', 'num__weight_kg', 'num__dose_mg', 'num__duration_days',
    'num__concurrent_meds', 'num__time_to_onset_days', 'num__comorbidity_count',
    'num__high_risk_age_flag', 'num__dose_weight_ratio', 'num__organ_impairment_score',
    'num__prior_ade',
    'cat__sex_Female', 'cat__sex_Male', 'cat__sex_Unknown',
    'cat__route_IV', 'cat__route_intramuscular', 'cat__route_oral',
    'cat__route_subcutaneous', 'cat__route_topical',
    'cat__indication_anticoagulation', 'cat__indication_cancer',
    'cat__indication_depression', 'cat__indication_diabetes',
    'cat__indication_epilepsy', 'cat__indication_hyperlipidemia',
    'cat__indication_hypertension', 'cat__indication_infection', 'cat__indication_pain',
    'cat__indication_transplant',
    'cat__reporter_type_consumer', 'cat__reporter_type_nurse',
    'cat__reporter_type_pharmacist', 'cat__reporter_type_physician',
    'cat__reaction_category_allergic', 'cat__reaction_category_cardiac',
    'cat__reaction_category_dermatological', 'cat__reaction_category_gastrointestinal',
    'cat__reaction_category_hematological', 'cat__reaction_category_hepatic',
    'cat__reaction_category_neurological', 'cat__reaction_category_renal',
    'cat__renal_function_mild_impairment', 'cat__renal_function_moderate_impairment',
    'cat__renal_function_normal', 'cat__renal_function_severe_impairment',
    'cat__hepatic_function_mild_impairment', 'cat__hepatic_function_moderate_impairment',
    'cat__hepatic_function_normal',
    'cat__drug_class_NSAID', 'cat__drug_class_antibiotic',
    'cat__drug_class_anticoagulant', 'cat__drug_class_antidepressant',
    'cat__drug_class_antidiabetic', 'cat__drug_class_antiepileptic',
    'cat__drug_class_antihypertensive', 'cat__drug_class_chemotherapy',
    'cat__drug_class_immunosuppressant', 'cat__drug_class_statin',
    'cat__drug_class_risk_tier_high', 'cat__drug_class_risk_tier_low',
    'cat__drug_class_risk_tier_medium', 'cat__drug_class_risk_tier_very_high',
    'cat__reporter_credibility_high', 'cat__reporter_credibility_low',
    'cat__reporter_credibility_medium',
    'cat__onset_speed_delayed', 'cat__onset_speed_moderate', 'cat__onset_speed_rapid',
    'cat__age_category_elderly', 'cat__age_category_middle_aged',
    'cat__age_category_young_adult',
    'cat__polypharmacy_risk_high', 'cat__polypharmacy_risk_low',
    'cat__polypharmacy_risk_moderate',
]


def scaled_data_to_classifier_format(scaled_data):
    values = scaled_data[0]
    return dict(zip(CLASSIFIER_SCHEMA, values))


def create_report_dataframe(age, sex, weight_kg, drug_class, dose_mg, route,
                            duration_days, indication, concurrent_meds, reporter_type,
                            time_to_onset_days, reaction_category, renal_function,
                            hepatic_function, comorbidity_count, prior_ade):
    raw_data = {
        'age':                age,
        'sex':                sex,
        'weight_kg':          weight_kg,
        'drug_class':         drug_class,
        'dose_mg':            dose_mg,
        'route':              route,
        'duration_days':      duration_days,
        'indication':         indication,
        'concurrent_meds':    concurrent_meds,
        'reporter_type':      reporter_type,
        'time_to_onset_days': time_to_onset_days,
        'reaction_category':  reaction_category,
        'renal_function':     renal_function,
        'hepatic_function':   hepatic_function,
        'comorbidity_count':  comorbidity_count,
        'prior_ade':          prior_ade,
    }
    df = pd.DataFrame([raw_data])
    return add_derived_features(df)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADE Risk Assessment",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1B5E82;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .serious-alert {
        background-color: #B71C1C;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #D32F2F;
    }
    .nonserious-alert {
        background-color: #1B5E20;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #2E7D32;
    }
    .warning-alert {
        background-color: #E65100;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #F57C00;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💊 Adverse Drug Event Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Pharmacovigilance Signal Evaluation — Powered by Domino Data Lab</p>',
            unsafe_allow_html=True)

# ── Model selection ───────────────────────────────────────────────────────────
st.subheader("Model Selection")
selected_model = st.selectbox(
    "Choose Classifier Model",
    options=list(model_scaling_dict.keys()),
    index=0
)

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Patient Demographics")
    age        = st.slider("Patient Age", 18, 95, 55)
    sex        = st.selectbox("Sex", ["Male", "Female", "Unknown"])
    weight_kg  = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)
    comorbidity_count = st.slider("Number of Comorbidities", 0, 10, 2)
    prior_ade  = st.selectbox("Prior ADE History", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    st.subheader("Drug Information")
    drug_class   = st.selectbox("Drug Class", [
        'NSAID', 'antibiotic', 'anticoagulant', 'antidepressant',
        'antidiabetic', 'antiepileptic', 'antihypertensive',
        'chemotherapy', 'immunosuppressant', 'statin'
    ])
    dose_mg      = st.number_input("Dose (mg)", min_value=0.1, max_value=2000.0, value=50.0, step=0.1)
    route        = st.selectbox("Route of Administration", ['oral', 'IV', 'subcutaneous', 'intramuscular', 'topical'])
    duration_days = st.number_input("Duration on Drug (days)", min_value=1, max_value=365, value=14)
    indication   = st.selectbox("Indication", [
        'pain', 'hypertension', 'diabetes', 'infection', 'depression',
        'cancer', 'transplant', 'hyperlipidemia', 'epilepsy', 'anticoagulation'
    ])

with col3:
    st.subheader("Event & Clinical Context")
    reaction_category = st.selectbox("Reaction Category", [
        'cardiac', 'hepatic', 'allergic', 'neurological',
        'gastrointestinal', 'dermatological', 'hematological', 'renal'
    ])
    time_to_onset_days = st.slider("Time to Onset (days)", 0, 90, 7)
    reporter_type      = st.selectbox("Reporter Type", ['physician', 'pharmacist', 'nurse', 'consumer'])
    concurrent_meds    = st.slider("Concurrent Medications", 0, 12, 3)
    renal_function     = st.selectbox("Renal Function", ['normal', 'mild_impairment', 'moderate_impairment', 'severe_impairment'])
    hepatic_function   = st.selectbox("Hepatic Function", ['normal', 'mild_impairment', 'moderate_impairment'])

st.markdown("---")
predict_button = st.button("Evaluate ADE Seriousness Risk", type="primary")

if predict_button:
    with st.spinner("Evaluating adverse event report..."):
        time.sleep(1)

        report_df = create_report_dataframe(
            age=age, sex=sex, weight_kg=weight_kg,
            drug_class=drug_class, dose_mg=dose_mg, route=route,
            duration_days=duration_days, indication=indication,
            concurrent_meds=concurrent_meds, reporter_type=reporter_type,
            time_to_onset_days=time_to_onset_days, reaction_category=reaction_category,
            renal_function=renal_function, hepatic_function=hepatic_function,
            comorbidity_count=comorbidity_count, prior_ade=prior_ade
        )

        report_json = {
            "data": {
                "age":                str(report_df['age'].iloc[0]),
                "sex":                report_df['sex'].iloc[0],
                "weight_kg":          str(report_df['weight_kg'].iloc[0]),
                "drug_class":         report_df['drug_class'].iloc[0],
                "dose_mg":            str(report_df['dose_mg'].iloc[0]),
                "route":              report_df['route'].iloc[0],
                "duration_days":      str(report_df['duration_days'].iloc[0]),
                "indication":         report_df['indication'].iloc[0],
                "concurrent_meds":    str(report_df['concurrent_meds'].iloc[0]),
                "reporter_type":      report_df['reporter_type'].iloc[0],
                "time_to_onset_days": str(report_df['time_to_onset_days'].iloc[0]),
                "reaction_category":  report_df['reaction_category'].iloc[0],
                "renal_function":     report_df['renal_function'].iloc[0],
                "hepatic_function":   report_df['hepatic_function'].iloc[0],
                "comorbidity_count":  str(report_df['comorbidity_count'].iloc[0]),
                "prior_ade":          str(report_df['prior_ade'].iloc[0]),
                "high_risk_age_flag": str(report_df['high_risk_age_flag'].iloc[0]),
                "dose_weight_ratio":  str(round(report_df['dose_weight_ratio'].iloc[0], 4)),
                "organ_impairment_score": str(report_df['organ_impairment_score'].iloc[0]),
                "drug_class_risk_tier":   report_df['drug_class_risk_tier'].iloc[0],
                "reporter_credibility":   report_df['reporter_credibility'].iloc[0],
                "onset_speed":            report_df['onset_speed'].iloc[0],
                "age_category":           report_df['age_category'].iloc[0],
                "polypharmacy_risk":      report_df['polypharmacy_risk'].iloc[0],
            }
        }

        scaled_data    = None
        model_response = None

        try:
            response = requests.post(
                feature_scaling_endpoint,
                auth=(feature_scaling_auth, feature_scaling_auth),
                json=report_json
            )
            if response.status_code == 200:
                scaled_data      = response.json()['result']
                classifier_input = scaled_data_to_classifier_format(scaled_data)

                selected_endpoint = model_scaling_dict[selected_model]['endpoint']
                selected_auth     = model_scaling_dict[selected_model]['auth']

                try:
                    classifier_response = requests.post(
                        selected_endpoint,
                        auth=(selected_auth, selected_auth),
                        json={"data": classifier_input}
                    )
                    if classifier_response.status_code == 200:
                        model_response = classifier_response.json()['result']
                    else:
                        st.error(f"Classifier API Error: {classifier_response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Classifier Connection Error: {e}")
            else:
                st.error(f"Scaling API Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")

        # ── Heuristic risk scoring (for demo purposes) ────────────────────────
        DRUG_CLASS_RISK_WEIGHT = {
            'chemotherapy': 0.9, 'immunosuppressant': 0.85, 'anticoagulant': 0.75,
            'antiepileptic': 0.65, 'antidepressant': 0.45, 'antihypertensive': 0.35,
            'antidiabetic': 0.40, 'antibiotic': 0.30, 'NSAID': 0.25, 'statin': 0.20
        }
        HIGH_RISK_REACTIONS = {'cardiac', 'hematological', 'hepatic', 'renal'}

        risk_factors = [
            age > 65,
            drug_class in ('chemotherapy', 'immunosuppressant', 'anticoagulant'),
            renal_function in ('moderate_impairment', 'severe_impairment'),
            hepatic_function == 'moderate_impairment',
            concurrent_meds >= 6,
            reaction_category in HIGH_RISK_REACTIONS,
            prior_ade == 1,
            time_to_onset_days <= 3,
        ]

        base_score        = DRUG_CLASS_RISK_WEIGHT.get(drug_class, 0.3)
        additional_score  = sum(risk_factors) * 0.07
        final_risk_score  = min(base_score + additional_score, 1.0)
        is_serious        = final_risk_score > 0.5

        # ── Display result ────────────────────────────────────────────────────
        if final_risk_score > 0.70:
            css_class = "serious-alert"
            icon      = "SERIOUS ADE — IMMEDIATE REVIEW REQUIRED"
        elif final_risk_score > 0.50:
            css_class = "warning-alert"
            icon      = "POTENTIAL SERIOUS ADE — REVIEW RECOMMENDED"
        else:
            css_class = "nonserious-alert"
            icon      = "NON-SERIOUS ADE — STANDARD MONITORING"

        st.markdown(f"""
        <div class="{css_class}">
            <h2>{icon}</h2>
            <h3>Seriousness Score: {final_risk_score:.1%}</h3>
            <p><strong>Drug Class:</strong> {drug_class} &nbsp;|&nbsp;
               <strong>Reaction:</strong> {reaction_category} &nbsp;|&nbsp;
               <strong>Model Used:</strong> {selected_model}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Risk factor breakdown ─────────────────────────────────────────────
        st.subheader("Risk Factor Breakdown")
        risk_labels = [
            "Age > 65 (geriatric risk)",
            "High-risk drug class",
            "Renal impairment (moderate/severe)",
            "Hepatic impairment (moderate)",
            "Polypharmacy (≥6 concurrent meds)",
            "High-risk reaction category",
            "Prior ADE history",
            "Rapid onset (≤3 days)"
        ]
        risk_df = pd.DataFrame({
            "Risk Factor": risk_labels,
            "Present":     ["Yes" if f else "No" for f in risk_factors],
            "Status":      ["⚠️" if f else "✅" for f in risk_factors]
        })
        st.dataframe(risk_df, use_container_width=True)

        st.subheader("Derived Feature Values")
        derived_df = pd.DataFrame({
            "Feature": [
                "High Risk Age Flag", "Dose/Weight Ratio", "Organ Impairment Score",
                "Drug Class Risk Tier", "Reporter Credibility",
                "Onset Speed", "Age Category", "Polypharmacy Risk"
            ],
            "Value": [
                str(report_df['high_risk_age_flag'].iloc[0]),
                f"{report_df['dose_weight_ratio'].iloc[0]:.4f}",
                str(report_df['organ_impairment_score'].iloc[0]),
                report_df['drug_class_risk_tier'].iloc[0],
                report_df['reporter_credibility'].iloc[0],
                report_df['onset_speed'].iloc[0],
                report_df['age_category'].iloc[0],
                report_df['polypharmacy_risk'].iloc[0],
            ]
        })
        st.dataframe(derived_df, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#546E7A;'>"
    "💊 Adverse Drug Event Risk Assessment v1.0 | Built with Streamlit on Domino Data Lab"
    "</p>",
    unsafe_allow_html=True
)
