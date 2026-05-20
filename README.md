# Genomic Prediction of Infliximab Response in IBD Patients

A translational bioinformatics and machine learning project designed to predict primary non-response to anti-TNF therapy (Infliximab) using baseline mucosal transcriptomic profiles from Inflammatory Bowel Disease (IBD) patients, validated across independent clinical cohorts.

---

## Project Overview
 * Infliximab is a highly effective biologic drug for treating Crohn’s Disease and Ulcerative Colitis (UC). However, 30–40% of patients suffer from primary non-response, where they receive zero therapeutic effect. This leads to months of unnecessary inflammation, disease progression, and massive healthcare costs during clinical trial-and-error.
 * Aimed to build a predictive binary classification model capable of screening a patient's baseline gut transcriptomic data before treatment starts to identify potential non-responders.
 * Data for model training was sourced from the NCBI GEO Database, Accession ID: **GSE16879**. Microarray gene expression profiling ($GPL570$ platform) consisting of 121 IBD patients and 54,675 genetic features.
 * Data for external validation was also sourced from NCBI GEO, Accession ID: **GSE23597**. A separate clinical cohort consisting of longitudinal sample points. Data was filtered to isolate 45 independent baseline (Week 0) patient profiles to test model generalization.

---

## Machine Learning & Validation Pipeline

The project is structured across two stages of deployment:

### Stage 1: Model Discovery & Feature Optimization (`02_final_pipeline.ipynb`)
1. Data Ingestion & Automation: Queries, downloads, and unpacks the raw zipped NCBI GEO series matrix files into local storage.
2. Clinical Target Engineering: Parsed sample-specific metadata, isolated the 121 target IBD patients (61 Ulcerative Colitis, 60 Crohn’s Disease), eliminated healthy controls, and engineered a clean binary tracking target (`Yes` / `No` response).
3. Genomic Alignment & Transposition: Rotated and filtered the gene expression matrix to align patient arrays with their respective clinical outcomes (Rows = Patients, Columns = Genes).
4. Validation Isolation (Train/Test Split): Implemented an 80/20 partition to lock away a 20% holdout validation set (25 unseen patients) before modeling, eliminating data leakage.
5. Modeling: Deployed an $L_1$-Penalized (Lasso) Logistic Regression solver (`liblinear`) to address the "Curse of Dimensionality" ($54,675 \text{ features} \gg 96 \text{ training samples}$) and avoid over-fitting.
6. Feature Optimization: Used the Lasso penalty to mathematically eliminate 97.7% of genes, resulting in 1,245 surviving predictive genes.
7. Model Serialization: Deployed model coefficients and serialized final pipeline assets using `joblib`.

### Stage 2: Independent External Validation (`04_multi_cohort_validation_final.ipynb`)
8. Parsing: Programmatically ingested the external validation matrix (**GSE23597**), removing metadata headers.
9. Filtering: Isolated  Week 0 baseline data (`Time == 'W0'`), removing Week 8 and Week 30 genomic metrics to prevent longitudinal data leakage.
10. Index-Based Inner Join: Elevated patient tracking codes (`geo_accession`) to the primary Index and ran a horizontal inner join to align the external clinical labels with the transposed feature matrix.
11. External Model Evaluation: Reloaded the serialized Lasso classifier using `joblib` and subjected it to an external validation exam on the 45 new patients.

---

## Performance Results & The Generalization Gap

* **Internal Validation Accuracy:** **72.00%** on the internal 20% holdout test set (GSE16879).
* **External Validation Accuracy:** **40.00%** on the independent replication cohort (GSE23597).

### Generalization Gap
The  drop from 72% internal accuracy to 40% external accuracy represents a classic hurdle in translational bioinformatics:
* Dimensionality vs. Sample Size: With 54,675 features and under 100 training samples, regularized models can  latch onto population-specific markers unique to the original cohort rather than universal anti-TNF biomarkers.
* Phenotypic Variance: Slight differences in how clinicians across different medical facilities define, score, and group "clinical response."

---

## Key Biological Discovery

The strongest positive predictor of therapeutic response calculated by the algorithm was probe **`214598_at`**, which maps directly to the human gene **CLDN8 (Claudin-8)**.

### Clinical Significance
Claudin-8 is a vital structural tight-junction protein responsible for maintaining the physical barrier integrity of the intestinal lining. The machine learning model independently discovered that baseline epithelial barrier robustness is a primary indicator for Infliximab therapeutic success. High baseline expression of CLDN8 strongly correlates with a positive clinical response.

---

## Repository Structure
* `01_exploratory_analysis.ipynb` - Initial data exploration, debugging, and experimental parsing scripts for the discovery cohort.
* `02_final_pipeline.ipynb` - The polished, streamlined pipeline used to engineer features and train the primary Lasso classifier.
* `03_multi_cohort_evaluation_draft.ipynb` - Scratchpad notebook dealing with longitudinal masking and string engineering bugs for the second dataset.
* `04_multi_cohort_validation_final.ipynb` - The clean, production-ready external validation pipeline executing the cross-cohort model evaluation.
* `infliximab_top_10_biomarkers.csv` - The top 10 calculated gene coefficients and names.
* `infliximab_lasso_model.pkl` - The trained, serialized machine learning model.
* `.gitignore` - Filters out local raw NCBI data folders to keep the repository lightweight.

---

## How to Run this Project Locally

### 1. Prerequisites
Ensure you have a Python environment (3.11+) activated and install the necessary dependencies:
```bash
pip install pandas numpy scikit-learn joblib