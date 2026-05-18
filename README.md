# Genomic Prediction of Infliximab Response in IBD Patients

A translational bioinformatics and machine learning pipeline designed to predict primary non-response to anti-TNF therapy (Infliximab) using baseline mucosal transcriptomic profiles from Inflammatory Bowel Disease (IBD) patients.

---

##  Project Overview
* Infliximab is a highly effective biologic drug for treating Crohn’s Disease and Ulcerative Colitis (UC). However, 30–40% of patients suffer from primary non-response, where they do not receive any therapeutic effect after starting treatmebt . This leads to months of unnecessary inflammation, disease progression, and massive healthcare costs during clinical trial-and-error.
*  I aimed to build a predictive binary classification model capable of screening a patient's baseline gut transcriptomic data before treatment starts to identify potential non-responders.
* Data was sourced from the NCBI GEO Database, Accession ID: GSE16879. Microarray gene expression profiling ($GPL570$ platform) consisting of 121 IBD patients and 54,675 genetic features.

---

## The Machine Learning Pipeline

The project is condensed into a17-cell pipeline structured across 7  phases:

1. **Data Ingestion & Automation:** Programmatically queries, downloads, and unpacks the raw zipped NCBI GEO series matrix files into local storage.
2. **Clinical Target Engineering:** Parsed sample-specific metadata, isolated the 121 target IBD patients (61 Ulcerative Colitis, 60 Crohn’s Disease), eliminated healthy controls, and engineered a clean binary tracking target (`Yes` / `No` response).
3. **Genomic Alignment & Transposition:** Rotated and filtered the gene expression matrix to align patient arrays with their respective clinical outcomes.
4. **Validation Isolation (Train/Test Split):** Implemented an 80/20 partition to lock away a 20% holdout validation set (25 unseen patients) before modeling, eliminating data leakage.
5. **Regularized Modeling:** Deployed an $L_1$-Penalized (Lasso) Logistic Regression solver (`liblinear`) to aggressively address the "Curse of Dimensionality" ($54,675 \text{ features} \gg 96 \text{ training samples}$) and avoid over-fitting.
6. **Feature Optimization:** Used the Lasso penalty to mathematically eliminate 97.7% of genes, resulting in **1,245 surviving predictive genes**.
7. **Model Serialization:** Deployed model coefficients and serialized the final pipeline assets using `joblib` for clinical software integration.

---

##  Performance Results

* **Validation Accuracy:** **72.00%** on completely unseen patient profiles.
* **Noise Reduction:** The model successfully identified that **53,430 genes** had zero predictive power regarding anti-TNF response and discarded them, retaining only the 1,245 core predictive signatures.

---

##  Key Biological Discovery

The strongest positive predictor of therapeutic response calculated by the algorithm was probe **`214598_at`**, which maps directly to the human gene **CLDN8 (Claudin-8)**.

### Clinical Significance
Claudin-8 is a vital structural tight-junction protein responsible for maintaining the physical barrier integrity of the intestinal lining. 

The machine learning model independently discovered that baseline epithelial barrier robustness is the primary predictor for Infliximab therapeutic success. High baseline expression of CLDN8 strongly correlates with a positive clinical response.

---

##  Repository Structure
* `01_exploratory_analysis.ipynb` — The initial scratchpad notebook detailing data exploration and debugging.
* `02_final_pipeline.ipynb` — The optimized 17-cell script.
* `infliximab_top_10_biomarkers.csv` — The top 10 calculated gene coefficients and names.
* `infliximab_lasso_model.pkl` — The trained model.
* `.gitignore` — Filters out the raw NCBI data matrices to keep the repository lightweight.

---

##  How to Run this Project Locally

### 1. Prerequisites
Ensure you have an Anaconda environment activated with Python 3.11+ and the required packages installed:
```bash
pip install pandas numpy scikit-learn joblib