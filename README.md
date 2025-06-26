# From Patterns to Predictions: A Multi-Model Exploration of Depression Risk

## Overview

This project explores depression risk in the U.S. adult population using data from the **2021–2023 National Health and Nutrition Examination Survey (NHANES)**. The team applied **unsupervised learning** to identify latent subgroups and **supervised modeling** to predict PHQ-9-based depression severity categories. The pipeline emphasizes model interpretability and integration of behavioral, medical, and socioeconomic factors. The final models combine data-driven clustering, dimensionality reduction, and classification performance evaluation using SHAP, confusion matrices, and multiple performance metrics.

---

## Team Members

- **Alexis Parker**  
- **Vikram Vaddamani**  
- **Brandon Fox**

This was a collaborative project. Responsibilities and contributions are detailed in the Statement of Work below.

---

## Repository Structure
```
depression-risk-modeling/
│
├── data/
│ ├── raw/ # Raw NHANES module CSVs (converted from .xpt)
│ ├── clean/ # Cleaned and merged datasets
│ └── external/ # Census-based reference data (e.g. income by demo group)
│
├── scripts/
│ ├── nhanes_download_convert.ipynb # Script to download and convert .xpt NHANES data
│ └── recover_seqn_demo_link.py # Recovery script to manage SEQN-based merges
│
├── sl/ # Supervised learning notebooks by team member and UL method
│ ├── KMEANStoSL_alexis.ipynb
│ ├── PCAtoSL_vik.ipynb
│ └── DBSCAN_to_SL_BFOX.ipynb
│
└── README.md # Project summary and instructions
```
---

## Data Sources

- **NHANES 2021–2023 Questionnaire Data**  
  Modules used: DEMO, DPQ, FNQ, HIQ, HUQ, INQ, SLQ  
  - Adults aged 18+ with valid physical and interview data  
  - PHQ-9 total scores calculated using standard criteria  
  - Final dataset: **467 rows × 26 cleaned features**

- **U.S. Census Reference Data**  
  Used to engineer estimated income per individual based on education level, race/ethnicity, and gender.

---

## Feature Engineering

- NHANES variables were cleaned, renamed, and filtered based on:
  - Conceptual relevance to depression (e.g., anxiety, sleep, insurance access)
  - Missingness thresholds
  - Interpretability and redundancy
- A **Census-derived estimated income** feature was created for each respondent using demographic group medians.

---

## Unsupervised Learning

Three methods were used to discover latent patterns and reduce dimensionality:

### 1. K-Means Clustering
- Preprocessing: one-hot encoding, median/mode imputation, standardization
- Final model: `k=4` (based on Elbow + Silhouette analysis)
- Identified groupings with distinct PHQ-9 scores and key variables such as:
  - Weekday sleep hours
  - Access to care
  - Poverty index
  - Household size

### 2. Principal Component Analysis (PCA)
- Top 20 components explained 95% of variance
- Key features included: weekday/weekend sleep, education level, age, access to care
- Used as input to supervised models

### 3. DBSCAN
- Noise-tolerant clustering
- Top 15 predictors selected based on cluster density
- Combined with engineered income for SL phase

---

## Supervised Learning

### Models Used:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

### Workflow:
- Target: **Binned PHQ-9 Severity**  
  ("None/Minimal", "Mild", "Moderate", "Moderately Severe", "Severe")

- Preprocessing:
  - Stratified train/test split
  - Imputation and standardization
  - GridSearchCV with Stratified K-Fold (k=5)

- Evaluation Metrics:
  - **Accuracy**
  - **Weighted F1-Score**
  - **Macro-Averaged ROC-AUC**
  - Confusion matrices
  - SHAP visualizations (Random Forest only)

---

## Key Results

| Model Source | Accuracy | F1-Score | ROC-AUC |
|--------------|----------|----------|---------|
| LogReg (DBSCAN) | 0.62 | 0.22 | 0.792 |
| RF (DBSCAN)     | **0.71** | **0.43** | **0.884** |
| SVM (DBSCAN)    | 0.63 | 0.22 | 0.785 |
| LogReg (PCA)    | 0.44 | 0.40 | 0.663 |
| RF (KMeans)     | 0.31 | 0.31 | 0.588 |

- **Top predictors** across models:  
  - Estimated Income  
  - Sleep Hours (Weekday and Weekend)  
  - Age  
  - Anxiety Frequency  
  - Access to Care Indicators

- **SHAP Analysis** highlighted the impact of sleep, anxiety, and socioeconomic context on misclassified cases.

---

## Failure & Sensitivity Analysis

- Severe cases were consistently underclassified.
- Removing depression-related features (sensitivity test) dropped model performance:
  - e.g., LogReg Accuracy ↓ from 0.44 to 0.36
- Trade-offs were observed between coverage (Random Forest) and calibration (Logistic Regression).

---

## Ethical Considerations

- PHQ-9 reliance does not capture full complexity of mental health
- Even anonymized demographic data may carry re-identification risk
- Causal misinterpretation (e.g., assuming predictors *cause* depression) must be avoided
- Clear communication of model limitations is essential for public trust

---

## Statement of Work

| Team Member | Contributions |
|-------------|---------------|
| **Alexis Parker** | GitHub setup, data merging, KMeans, supervised modeling (LogReg, RF, SVM), SHAP, visualization, writing |
| **Brandon Fox**   | DBSCAN, SL models, evaluation, SHAP, Census joining, writing |
| **Vikram Vaddamani** | PCA, SL modeling, seaborn visualizations, tuning, writing |

---

## How to Reproduce

1. **Download & convert NHANES data**  
   Run `scripts/nhanes_download_convert.ipynb` to pull and convert data into `/data/raw`.

2. **Clean and merge modules**  
   Use scripts to generate `/data/clean/merged_clean.csv`.

3. **Run supervised notebooks**  
   Choose from:
   - `sl/KMEANStoSL_alexis.ipynb`
   - `sl/PCAtoSL_vik.ipynb`
   - `sl/DBSCAN_to_SL_BFOX.ipynb`

4. **Examine SHAP and Confusion Matrices**  
   Review model explanations and classification results.

---

## References

- Vu et al. (2025) – *Prediction of depressive disorder using ML, NHANES*  
- Kim (2025) – *Socioeconomic perspectives on depression, KNHANES*  
- Califf et al. (2022) – *Social determinants and PHQ-9 screening*
