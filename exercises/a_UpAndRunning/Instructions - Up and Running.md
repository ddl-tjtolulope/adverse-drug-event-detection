# Exercise 1: Up and Running

## Objective
Set up your Domino project, configure team collaboration, and familiarise yourself with the platform governance workflows.

---

## Steps

### 1. Create Your Project
1. Log into your Domino instance
2. Click **New Project** from the home screen
3. Name it `ADE-Detection-Workshop-<your-initials>`
4. Select the **Adverse Drug Event Detection** project template if available, otherwise start from scratch

### 2. Import This Repository
1. In your new project, navigate to **Settings → Git Repositories**
2. Add this repository: `https://github.com/ddl-tjtolulope/adverse-drug-event-detection`
3. Confirm the repository is visible under **Files**

### 3. Configure the Data Source
1. Navigate to **Data → Data Sources**
2. Create a new S3 (or equivalent) data source named `ade_reports`
3. Upload `raw_ade_reports.csv` to this data source
   - If you don't have the FAERS CSV, run `python generate_dataset.py` locally to generate synthetic data

### 4. Set Up Collaborators
1. Go to **Project Settings → Collaborators**
2. Add at least one team member with **Contributor** access
3. Review the access control options (Viewer, Contributor, Owner)

### 5. Create a Governed Bundle (Optional)
1. Navigate to **Govern → Bundles**
2. Create a new bundle named `ADE-Workshop-v1`
3. Associate it with this project
4. Submit for approval to experience the governance workflow

---

## Key Concepts Demonstrated
- **Project Templates** — accelerate project setup with standardised configurations
- **Git Integration** — connect to external repositories for version-controlled code
- **Data Sources** — secure, driver-free connections to external data stores
- **Governed Bundles** — audit-ready packaging for regulated environments (critical in pharma)

---

## Relevant Documentation
- [Work with Projects](https://docs.dominodatalab.com/en/cloud/user_guide/a8e081/work-with-projects/)
- [Create Governed Bundles](https://docs.dominodatalab.com/en/cloud/user_guide/d56edd/create-governed-bundles/)
- [Project Templates](https://docs.dominodatalab.com/en/cloud/user_guide/5fed45/project-templates/)
