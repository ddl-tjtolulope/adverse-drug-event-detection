# Exercise 2: Data Exploration

## Objective
Use a Jupyter workspace on Domino to explore the adverse drug event dataset, perform data cleaning, and generate visualisations of seriousness patterns across drug classes, patient demographics, and reaction types.

---

## Steps

### 1. Start a Jupyter Workspace
1. From your project, click **Workspaces → New Workspace**
2. Select **JupyterLab** as the IDE
3. Choose a hardware tier appropriate for data exploration (e.g., Small — 2 vCPU / 4 GB RAM)
4. Launch the workspace

### 2. Open the Notebook
1. Navigate to `exercises/b_DataExploration/data_exploration_notebook.ipynb`
2. Run all cells sequentially (Kernel → Restart & Run All)

### 3. Connect to the Data Source
The notebook will connect to the `ade_reports` Domino Data Source you configured in Exercise 1. It will:
- Download `raw_ade_reports.csv` from the data source
- Load it into a Pandas DataFrame

### 4. Review the Exploratory Analysis
The notebook covers:
- **Dataset Overview** — shape, memory, column types
- **Data Cleaning** — identifying and removing rows with missing values
- **Seriousness Rate Analysis** — overall and by drug class, reaction category, age group
- **Feature Distributions** — dose, duration, concurrent medications
- **Correlation Analysis** — which features correlate most with serious outcomes
- **Data Quality Report** — missing values, outliers, class balance

### 5. Save the Clean Dataset
The final cell saves `clean_ade_reports.csv` to your Domino Dataset for use in Exercise 3.

---

## Key Concepts Demonstrated
- **Domino Workspaces** — reproducible, cloud-hosted Jupyter environments
- **Data Source Connectors** — secure data access without managing credentials
- **Domino Datasets** — versioned, persistent data storage shared across executions
- **EDA in Life Sciences** — understanding data quality is critical before building safety models

---

## Relevant Documentation
- [Start a Jupyter Workspace](https://docs.dominodatalab.com/en/cloud/user_guide/93aef2/start-a-jupyter-workspace/)
