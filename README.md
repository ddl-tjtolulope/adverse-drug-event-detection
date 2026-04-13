# Adverse Drug Event Detection Workshop

This workshop provides hands-on experience with the Domino Data Lab platform while completing the full model development and delivery lifecycle — from raw pharmacovigilance data through production deployment of an adverse drug event (ADE) risk scoring system.

---

## Workshop Overview

Adverse Drug Events (ADEs) are a leading cause of hospital admissions and represent a critical area of focus for pharmaceutical companies, CROs, and regulators. This workshop walks through building an ML-based system to classify whether a drug safety report represents a **serious** adverse event — directly mirroring real-world pharmacovigilance workflows.

The following exercises are designed to be completed sequentially. Each exercise directory contains all necessary resources including instructions, notebooks, and scripts.

---

### Exercise 1: Platform Setup
Project initialization, team collaboration setup, and governance workflow approval.

**Instructions:** [Up and Running Guide](exercises/a_UpAndRunning/Instructions%20-%20Up%20and%20Running.md)
**Relevant Documentation:** [Work with Projects](https://docs.dominodatalab.com/en/cloud/user_guide/a8e081/work-with-projects/) | [Create Governed Bundles](https://docs.dominodatalab.com/en/cloud/user_guide/d56edd/create-governed-bundles/) | [Project Templates](https://docs.dominodatalab.com/en/cloud/user_guide/5fed45/project-templates/)

---

### Exercise 2: Data Exploration
Interactive data analysis using Jupyter notebooks. Import adverse event report data from the FDA FAERS-inspired dataset, perform data cleaning, generate visualizations of seriousness patterns across drug classes and patient populations, and save the processed dataset.

**Instructions:** [Data Exploration Guide](exercises/b_DataExploration/Instructions%20-%20Data%20Exploration.md)
**Relevant Documentation:** [Start a Jupyter Workspace](https://docs.dominodatalab.com/en/cloud/user_guide/93aef2/start-a-jupyter-workspace/)

---

### Exercise 3: Data Engineering
Automated data processing pipeline using batch jobs. Apply pharmacovigilance-specific feature engineering (organ impairment scoring, drug class risk tiers, polypharmacy risk), data normalization, and scaling transformations. Generate versioned datasets and preprocessing models.

**Instructions:** [Data Engineering Guide](exercises/c_DataEngineering/Instructions%20-%20Data%20Engineering.md)
**Relevant Documentation:** [Create and run Jobs](https://docs.dominodatalab.com/en/latest/user_guide/af97b7/create-and-run-jobs/)

---

### Exercise 4: Model Training and Evaluation
Orchestrated model training using Domino Flows. Execute parallel training workflows for XGBoost, AdaBoost, and Gaussian Naive Bayes classifiers, compare performance metrics (ROC-AUC, F1, calibration), and register the optimal model for deployment.

**Instructions:** [Training and Evaluation Guide](exercises/d_TrainingAndEvaluation/Instructions%20-%20Training%20And%20Evaluation.md)
**Relevant Documentation:** [Define Flows](https://docs.dominodatalab.com/en/latest/user_guide/e09156/define-flows/) | [Track and Monitor Experiments](https://docs.dominodatalab.com/en/cloud/user_guide/da707d/track-and-monitor-experiments/) | [Manage Models with Model Registry](https://docs.dominodatalab.com/en/cloud/user_guide/3b6ae5/manage-models-with-model-registry/)

---

### Exercise 5: Model Deployment
Production model deployment through multiple channels:
- REST API endpoints for real-time ADE seriousness scoring
- Interactive Streamlit web application for safety report review and model comparison
- Secure configuration management for production deployment

**Instructions:** [Hosting and Execution Guide](exercises/e_HostingAndExecution/Instructions%20-%20HostingAndExecution.md)
**Relevant Documentation:** [Deploy your Python model](https://docs.dominodatalab.com/en/latest/user_guide/9f10c9/deploy-your-python-model/)

---

## Dataset

This workshop uses an FDA FAERS-inspired adverse drug event dataset (~80,000 reports). The dataset includes:
- Patient demographics (age, sex, weight)
- Drug information (name, class, dose, route, duration)
- Clinical context (indication, concurrent medications, organ function)
- Event characteristics (reaction category, time to onset, reporter type)
- **Target variable:** `Serious` (1 = serious adverse event, 0 = non-serious)

**Source:** [FDA Adverse Event Reporting System (FAERS) — Kaggle](https://www.kaggle.com/datasets/fda/adverse-events)

A synthetic data generator (`generate_dataset.py`) is also provided for local development without a Kaggle account.
