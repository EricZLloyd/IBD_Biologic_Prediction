# Predicting Infliximab Response in IBD

This project tests whether gene expression in pre-treatment intestinal biopsies can predict response to infliximab in patients with inflammatory bowel disease (IBD). It trains a logistic regression model on one public cohort and evaluates it on a separate cohort.

Internal cross-validation accuracy was **68.97%**, compared with a **54.1%** majority-class baseline. External validation did not establish reliable predictive performance, and the selected probes varied substantially across training folds. The analysis does not support a validated biomarker or clinical prediction tool.

An earlier analysis included repeated samples from the same patients and reported an unsupported CLDN8 biomarker claim. Those results have been withdrawn; see the [correction notice](#correction-notice).

## Data

Both datasets are public NCBI GEO expression series. The analysis uses the downloaded series matrices, with 54,675 microarray probes as input features.

| Cohort | Role | Baseline patients | Response labels |
|---|---|---:|---|
| GSE16879 | Model development | 61 | 28 responders, 33 non-responders |
| GSE23597 | External validation | 45 | 30 responders, 15 non-responders |

GSE16879 contains 121 IBD biopsy samples: 61 before treatment and 60 after treatment. Only pre-treatment samples are retained. GSE23597 is restricted to Week 0 samples, with response labels assessed at Week 8.

## Methods

The development notebook, `02_final_pipeline.ipynb`:

1. Downloads the expression matrix, parses clinical metadata, removes healthy controls and retains baseline samples.
2. Aligns expression profiles with response labels using GEO sample accessions.
3. Fits L1-penalised logistic regression (`liblinear`) with `StandardScaler` inside a scikit-learn pipeline. Scaling is fitted within each training fold.
4. Estimates accuracy using nested, stratified cross-validation: three inner folds tune regularisation strength (`C`), and five outer folds evaluate the tuned models.
5. Counts how often each probe is selected across the five outer training folds.
6. Tunes and refits a final pipeline on all 61 baseline patients, then saves the model and its non-zero coefficients.

The tuning grid is `C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]`. The final fit selects `C = 100` and retains 704 probes.

The external validation notebook, `04_multi_cohort_validation_final.ipynb`, loads the saved model and evaluates it on GSE23597. It also runs an exploratory comparison in which the external cohort is standardised using its own means and standard deviations while the classifier remains fixed.

## Results

### Internal validation

| Metric | Result |
|---|---:|
| Mean accuracy across five outer folds | 68.97% |
| Standard deviation across folds | 9.13 percentage points |
| Majority-class baseline | 54.1% |

The outer-fold accuracies are 61.54%, 66.67%, 58.33%, 75.00% and 83.33%. These results suggest predictive signal within the development cohort, but each test fold contains only 12–13 patients. The fold standard deviation describes variation between folds; it is not a confidence interval.

### External validation

| Scaling approach | Predicted responders / non-responders | Accuracy | ROC AUC |
|---|---:|---:|---:|
| Saved development-cohort scaler | 45 / 0 | 66.67% | 0.642 |
| Scaler fitted to the external cohort | 24 / 21 | 55.56% | 0.627 |

With the saved pipeline, every patient is classified as a responder. Its 66.67% accuracy therefore equals the external cohort's majority-class baseline. The AUC reflects variation in predicted probabilities despite the identical class predictions.

Standardising the external cohort separately produces both response classes, but accuracy falls below the baseline. For this exploratory approach, the bootstrap 95% confidence interval for AUC is **0.424–0.804** (2,000 resamples), which includes 0.5. This interval applies to the AUC of 0.627, not the saved pipeline's AUC of 0.642.

The second approach uses the external cohort to estimate preprocessing parameters, so it is a cohort-level adaptation experiment. It does not evaluate the saved pipeline unchanged or show how it would perform on a single new patient.

## Interpretation and limitations

**Probe selection is unstable.** Across the five outer training folds, 3,322 distinct probes are selected, but only four appear in every fold: `1552726_at`, `1568787_at`, `203349_s_at` and `225807_at`. This limits the biological interpretation of individual coefficients. Neither a large coefficient in the final fit nor selection in all five folds establishes a biomarker.

**The cohorts have different expression distributions.** After applying the development-cohort scaler, the external expression matrix has an overall mean of 6.566 and standard deviation of 5.552. All external decision scores are positive, explaining the all-responder predictions. Standardising the external cohort separately restores mixed predictions without improving AUC in this comparison. This suggests a preprocessing or distribution mismatch, but does not establish its cause or rule out other approaches to batch correction.

**The sample sizes limit the conclusions.** There are 54,675 candidate probes and only 61 development patients. External validation includes 45 patients, leaving substantial uncertainty around performance. Differences in patient populations, response definitions and assessment timing may also affect transfer between cohorts; this analysis does not separate their effects.

The current results do not establish reliable prediction in an independent cohort or a reproducible gene signature. Larger cohorts and closer assessment of clinical and technical differences would be needed to evaluate those questions.

## Correction notice

The original development analysis used all 121 IBD samples in GSE16879, including pre- and post-treatment biopsies from the same patients. Splitting these samples without grouping by patient allowed the same patient to appear in both training and test data. Including post-treatment expression also conflicted with the aim of predicting response before treatment.

The corrected analysis retains only the 61 baseline profiles and uses nested cross-validation with scaling inside the pipeline. The previous **72% internal accuracy** and **40% external accuracy** are superseded by the results above.

The previous claim that **CLDN8** was the strongest response biomarker is also withdrawn. It came from the earlier model, and its probe (`214598_at`) is not retained in the corrected final fit. The analysis does not support the earlier biological interpretation of CLDN8.

## Repository contents

| File | Purpose |
|---|---|
| `02_final_pipeline.ipynb` | Current development pipeline, nested cross-validation, probe stability analysis and model export |
| `04_multi_cohort_validation_final.ipynb` | Current external validation and scaling comparison |
| `01_exploratory_analysis.ipynb` | Earlier exploratory work; superseded |
| `03_multi_cohort_evaluation_draft.ipynb` | Earlier external-validation draft; superseded |
| `infliximab_coefficients.csv` | The final model's 704 non-zero probe coefficients and outer-fold selection counts (`folds_selected`, 0–5) |
| `infliximab_lasso_model.pkl` | Saved scikit-learn pipeline containing the scaler and classifier |
| `data/` | Downloaded GEO data; excluded from version control |

## Running the analysis

Use Python 3.11+ and install the dependencies:

```bash
python -m pip install pandas numpy scikit-learn joblib jupyterlab
