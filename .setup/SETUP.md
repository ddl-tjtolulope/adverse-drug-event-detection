# Adverse Drug Event Detection Workshop — Admin Setup Guide

The following instructions guide you through deploying and configuring a Domino environment for the ADE Detection Workshop. These assume working knowledge of the platforms involved and do not include screenshots.

Anywhere you see an upper-case word in **BOLD** that is referring to a button that needs to be clicked.

---

## 1. Set Up the Domino Environment

- [ ] Create a Fleetcommand Domino instance as prescribed in `./fleetcommand.md`
- [ ] Create a copy of the setup checklist and work through it
- [ ] Create the Domino Compute Environment as per `./environment.md`

---

## 2. Generate and Upload Dataset to S3

The workshop uses an FDA FAERS-inspired synthetic dataset. Run the upload script to generate and push `raw_ade_reports.csv` to your S3 bucket:

```bash
pip install boto3

python .setup/upload_to_s3.py \
  --bucket  <your-s3-bucket-name> \
  --region  us-east-2 \
  --prefix  ade-workshop/
```

This will:
1. Generate ~80,000 synthetic ADE reports
2. Upload `raw_ade_reports.csv` to `s3://<bucket>/ade-workshop/raw_ade_reports.csv`

AWS credentials must be available via environment variables, `~/.aws/credentials`, or an IAM role.

---

## 3. Configure Data Source (Domino Admin Section)

**CREATE DATA SOURCE**

- [ ] Select Data Source: `Amazon S3`
- [ ] Bucket: `<your-s3-bucket-name>`
- [ ] Region: `us-east-2`
- [ ] Data Source Name: `adverse-drug-event-detection`
- [ ] Data Source Description:
  ```
  FDA FAERS-inspired adverse drug event dataset for the ADE Detection Workshop.
  Contains ~80,000 synthetic safety reports with patient demographics, drug information,
  reaction categories, and seriousness labels.
  ```

**NEXT**

- [ ] Credential Type: Select `Service Account`

**NEXT**

- [ ] If Nexus Data Planes: Select `Select All`

**NEXT**

- [ ] Access Key ID: Enter `<Your Access Key ID>`
- [ ] Secret Access Key: Enter `<Your Secret Access Key>`

**TEST CREDENTIALS**

- [ ] Update permissions by selecting the `Everyone` Radio Button

**FINISH SETUP**

---

## 4. Other Admin Configurations

- [ ] Billing Tags: `Pharma Safety Pharmacovigilance LifeSciences MLOps ADE`

---

## 5. Create Donor Project

**Create Project**

- [ ] Template: `None`
- [ ] Project Name: `ADE-Detection-Workshop-Donor`
- [ ] Visibility: `Public`

**NEXT**

- [ ] Hosted By: `Git Service Provider`
- [ ] Git Service Provider: `GitHub`
- [ ] Git Credentials: `None`
- [ ] Git Repo URL: `https://github.com/ddl-tjtolulope/adverse-drug-event-detection.git`

**CREATE**

- [ ] Set Default Compute Environment: `ADE-Detection-Workshop`
- [ ] Add Data Source: `adverse-drug-event-detection`
- [ ] Add Tags: `Pharmacovigilance, Adverse Drug Events, Drug Safety, Python, XGBoost, AdaBoost, GaussianNB, API, App`

---

## 6. Create Template From Donor Project

**CREATE TEMPLATE**

- [ ] Template Name: `ADE-Detection-Workshop-Template`
- [ ] Description: `Adverse Drug Event Detection Workshop — full lifecycle from raw safety reports to deployed ADE risk scoring API`
- [ ] Access: `Anyone with access...`

**NEXT**

- [ ] Ensure `Select All` is selected and deselect the following:
  - Goals
  - Datasets
  - External Volumes
  - Artifacts
  - Imported Projects
  - Published Entities
  - Integrations
- [ ] Default Billing Tag: `ADE`
- [ ] Default Environment: `ADE-Detection-Workshop`
- [ ] Default Hardware Tier: `Small`

**NEXT**

- [ ] File Storage: `In new Repo...`
- [ ] Git Service Provider: `GitHub`
- [ ] Git Credentials: `<Admin Workshop Credentials>`
- [ ] Owner: `<Select Owner>`
- [ ] Repo Visibility: `Public`

**CREATE**
