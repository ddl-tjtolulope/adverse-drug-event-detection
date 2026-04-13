# Exercise 5: Hosting and Execution

## Objective
Deploy the preprocessing pipeline and trained classifiers as REST API endpoints, then launch an interactive Streamlit application for real-time adverse drug event seriousness scoring.

---

## Steps

### 1. Deploy the Preprocessing Pipeline as a Model API
1. Navigate to **Model APIs → New Model API**
2. Name it `ADE-Feature-Scaling`
3. Set the file to `exercises/c_DataEngineering/data_engineering.py`
4. Set the function to `pipeline.transform`
5. Select the hardware tier and environment
6. Click **Deploy**
7. Note the **Endpoint URL** and **API Key** — you'll need these shortly

### 2. Deploy Each Classifier as a Model API
Repeat the above for each of the three trained classifiers:

| Model | File | Function |
|---|---|---|
| XGBoost | `exercises/d_TrainingAndEvaluation/trainer_xgb.py` | `model.predict_proba` |
| AdaBoost | `exercises/d_TrainingAndEvaluation/trainer_ada.py` | `model.predict_proba` |
| GaussianNB | `exercises/d_TrainingAndEvaluation/trainer_gnb.py` | `model.predict_proba` |

Alternatively, deploy models directly from the **Model Registry** (recommended for production).

### 3. Configure the Streamlit App
1. Copy the config template:
   ```
   cp exercises/e_HostingAndExecution/app_config_template.py \
      exercises/e_HostingAndExecution/app_config.py
   ```
2. Edit `app_config.py` and fill in the endpoint URLs and auth tokens from Steps 1–2
3. **Note:** `app_config.py` is in `.gitignore` — never commit credentials

### 4. Launch the Streamlit App
Run the app as a Domino App:
1. Navigate to **Apps → New App**
2. Set the command to:
   ```
   bash app.sh
   ```
3. Launch the app
4. The console will display the Streamlit URL

Or run locally:
```bash
PORT=8501 bash app.sh
```

### 5. Test the Application
Use the app to submit an adverse drug event report with these high-risk characteristics and verify the model flags it as serious:
- Age: 72, Drug Class: anticoagulant, Reaction: cardiac
- Renal Function: moderate_impairment, Concurrent Medications: 8
- Reporter: physician, Time to Onset: 2 days

---

## Key Concepts Demonstrated
- **Model API Deployment** — one-click REST API from a trained model
- **Multi-Model Architecture** — preprocessing + classifier as separate, composable APIs
- **Streamlit on Domino** — interactive web app for non-technical stakeholders
- **Secure Configuration** — credentials managed outside version control

---

## Relevant Documentation
- [Deploy your Python model](https://docs.dominodatalab.com/en/latest/user_guide/9f10c9/deploy-your-python-model/)
