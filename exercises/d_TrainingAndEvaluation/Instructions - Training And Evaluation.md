# Exercise 4: Model Training and Evaluation

## Objective
Use Domino Flows to orchestrate parallel training of three classifiers (XGBoost, AdaBoost, Gaussian Naive Bayes), compare their performance on the ADE seriousness classification task, and register the best model.

---

## Steps

### 1. Review the Training Scripts
Three trainer scripts each call the shared `generic_trainer.py`:

| Script | Model | Notes |
|---|---|---|
| `trainer_xgb.py` | XGBoost | Gradient boosting, handles feature interactions well |
| `trainer_ada.py` | AdaBoost | Ensemble boosting, good baseline |
| `trainer_gnb.py` | GaussianNB | Probabilistic, fast, good interpretability |

### 2. Open the Flow Notebook
1. Open `exercises/d_TrainingAndEvaluation/domino_flow_training_workflow.ipynb`
2. This notebook launches the Domino Flow defined in `workflow.py`
3. The flow runs all three trainers **in parallel**, then runs the comparison step

### 3. Launch the Flow
Run the flow notebook cells to trigger the Domino Flow. You can also run it manually:
```
python exercises/d_TrainingAndEvaluation/workflow.py
```

Monitor the parallel execution in **Flows → Runs**.

### 4. Compare Model Performance
Each trainer logs the following metrics to MLflow:

| Metric | Description |
|---|---|
| `roc_auc` | Area under the ROC curve (primary metric) |
| `pr_auc` | Average precision — important for class-imbalanced ADE data |
| `f1_serious` | F1 score for the serious class |
| `recall_serious` | Recall for serious events — key for safety applications |
| `ece` | Expected Calibration Error — probability reliability |
| `ks` | Kolmogorov-Smirnov statistic |

Navigate to **Experiments** and compare runs across the three models.

### 5. Review Model Artifacts
Each model produces three diagnostic charts:
- **Calibration Plot** — how well predicted probabilities match observed outcomes
- **Learning Curves** — accuracy, AUC, training time, and overfitting diagnostic
- **Performance Dashboard** — ROC curve, Precision-Recall, Confusion Matrix, threshold sweep

### 6. Register the Best Model
1. Identify the best model by ROC-AUC (the `compare.py` step outputs this)
2. Navigate to **Experiments → [best run] → Register Model**
3. Register it under the name `ADE-Seriousness-Classifier`
4. Tag it with `version: 1.0` and `stage: staging`

---

## Key Concepts Demonstrated
- **Domino Flows** — orchestrated, parallel ML workflows (Flyte-based)
- **MLflow Experiment Tracking** — side-by-side model comparison
- **Model Registry** — versioned model management for regulated environments
- **Calibration Evaluation** — critical in drug safety (probability outputs drive clinical decisions)

---

## Relevant Documentation
- [Define Flows](https://docs.dominodatalab.com/en/latest/user_guide/e09156/define-flows/)
- [Track and Monitor Experiments](https://docs.dominodatalab.com/en/cloud/user_guide/da707d/track-and-monitor-experiments/)
- [Manage Models with Model Registry](https://docs.dominodatalab.com/en/cloud/user_guide/3b6ae5/manage-models-with-model-registry/)
