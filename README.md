# Genomic Prediction of Infliximab Response in IBD Patients

A translational bioinformatics and machine learning project attempting to predict primary response to anti-TNF therapy (Infliximab) from baseline mucosal transcriptomic profiles, tested against an independent clinical cohort.

**Headline result: a negative one.** The classifier learns real signal within its discovery cohort (nested CV accuracy **68.97% ± 9.13%** against a 54.1% majority-class baseline), but does not transfer to an independent cohort (external AUC **0.627**, 95% CI **0.424–0.804** — not distinguishable from chance at n=45). The feature selection is unstable enough within the discovery cohort that the model has **no well-defined "top biomarker"** to report.

An earlier version of this project reported 72% internal / 40% external accuracy and a headline CLDN8 biomarker finding. **Both were artefacts of a data leakage bug and have been retracted.** See [Correction notice](#correction-notice).

---

## Project Overview

* Infliximab is a highly effective biologic for Crohn's Disease and Ulcerative Colitis. However, 30–40% of patients suffer primary non-response, receiving no therapeutic effect — months of unnecessary inflammation, disease progression, and cost during clinical trial-and-error.
* The aim was a binary classifier screening a patient's baseline gut transcriptomic profile *before* treatment starts, to identify likely non-responders.
* **Discovery cohort:** NCBI GEO **GSE16879**. Microarray expression profiling (GPL570), 54,675 probes. The series contains 121 IBD *samples* — pre- and post-treatment biopsies from the same individuals. After filtering to baseline (pre-treatment) samples only, **61 independent patients** remain.
* **External validation cohort:** NCBI GEO **GSE23597**, a separate longitudinal study filtered to **45 independent baseline (Week 0)** profiles.

---

## Machine Learning & Validation Pipeline

### Stage 1: Model discovery (`02_final_pipeline.ipynb`)

1. **Data ingestion:** queries, downloads and unpacks the raw zipped GEO series matrix files.
2. **Clinical target engineering:** parses sample metadata, removes healthy controls, and engineers a binary `Yes`/`No` response target.
3. **Baseline filtering (leakage guard):** filters to `before or after first infliximab treatment == 'Before first infliximab treatment'`, reducing 121 samples to 61 unique patients. Without this, roughly half the cohort appears twice — once pre- and once post-treatment — carrying the same patient-level response label. See [Correction notice](#correction-notice).
4. **Genomic alignment:** transposes and filters the expression matrix to align patient arrays with clinical outcomes (rows = patients, columns = probes).
5. **Modelling:** an $L_1$-penalised (Lasso) logistic regression (`liblinear`), chosen to address the curse of dimensionality (54,675 features ≫ 61 samples) via aggressive feature elimination.
6. **Per-fold standardisation:** `StandardScaler` and the classifier are wrapped in a `Pipeline`, so the scaler is fitted on training folds only. Unstandardised L1 is biased toward high-variance probes, because a low-variance probe needs a larger coefficient — and therefore a larger penalty — to exert the same effect.
7. **Nested cross-validation:** an inner 3-fold `GridSearchCV` tunes `C` over `[0.001 … 10000]`; an outer 5-fold `StratifiedKFold` scores each tuned model on data the inner loop never saw. Stratification matters at this sample size — a plain split could produce a badly skewed test fold from only ~12 patients.
8. **Selection stability analysis:** refits across the outer folds and counts how often each probe survives, testing whether the selected signature is reproducible at all.
9. **Model serialisation:** the deployable model is fitted separately on all 61 patients (nested CV yields an *estimate*, not a model) and serialised with `joblib`.

**Chosen `C`: 100** (interior to the grid — not clipped at a boundary). **704 probes survive** the Lasso penalty (98.7% eliminated).

### Stage 2: Independent external validation (`04_multi_cohort_validation_final.ipynb`)

10. **Parsing:** ingests the GSE23597 series matrix, stripping metadata headers.
11. **Filtering:** isolates Week 0 baseline data (`Time == 'W0'`), removing Week 8 and Week 30 samples to prevent longitudinal leakage.
12. **Index-based inner join:** promotes `geo_accession` to the index and inner-joins clinical labels to the transposed feature matrix.
13. **External evaluation:** reloads the serialised pipeline and scores it on the 45 unseen patients under two scaling regimes, with a bootstrap confidence interval on AUC.

---

## Results

### Internal (GSE16879, nested CV, n=61)

| Metric | Value |
|---|---|
| Nested CV accuracy | **68.97% ± 9.13%** |
| Per-fold scores | 0.615, 0.667, 0.583, 0.750, 0.833 |
| Class balance | 33 No / 28 Yes |
| Majority-class baseline | **54.1%** |

The model clears the trivial baseline by ~15 points, so it is learning something real within this cohort. The fold-to-fold spread is wide by construction: each outer test fold holds only ~12 patients. The mean is the estimate; individual folds should not be read.

**Robustness check:** before per-fold standardisation was added, nested CV returned 70.38% ± 11.50% at `C = 0.01` with 89 surviving genes. The performance difference (1.4 points) sits well inside the fold variance — standardisation did not change the result, it made the *feature selection* defensible. Note that `C` is not comparable across the two runs: standardising rescales every coefficient, and with it the penalty budget.

### External (GSE23597, n=45)

Class balance: **30 Yes / 15 No** → majority-class baseline **66.67%**.

| Scaling regime | Predictions | Accuracy | AUC |
|---|---|---|---|
| Scaler fitted on GSE16879 (strict unseen-data) | **Yes ×45, No ×0** | 66.67% | 0.642 |
| Scaler re-fitted on GSE23597 (per-cohort centring) | Yes ×24, No ×21 | 55.56% | 0.627 |

**External AUC 95% CI (bootstrap, 2000 resamples): 0.424 – 0.804.** The interval contains 0.5.

**Accuracy is uninformative here and is not the headline.** The 66.67% figure is the majority-class baseline hit by accident — the model predicted `Yes` for every single patient and scored 30/45 by doing nothing. The 55.56% figure comes from genuine mixed predictions but sits ~11 points *below* the baseline. Neither number carries information. AUC does, because it is threshold-independent.

---

## The feature selection is unstable

The single final fit produces a ranked coefficient list, and it is tempting to read the top of it as a discovery. That is precisely how the retracted CLDN8 finding was produced. So the selection was tested directly: refit across the five outer folds, and count how often each probe survives.

| | |
|---|---|
| Probes selected per fold | ~704 |
| **Union across 5 folds** | **3,322 distinct probes** |
| **Intersection (all 5 folds)** | **4 probes** — `1552726_at`, `1568787_at`, `203349_s_at`, `225807_at` |
| Top-weighted probe in the final fit (`216242_x_at`) | selected in **3 of 5** folds |

Five folds make ~3,500 selections in total, and they union to 3,322 — **almost every selection is a first appearance.** The folds pick near-disjoint gene sets from the same 61 patients; change which ~12 patients are held out and the signature reshuffles. Coefficient magnitude and selection stability are close to decoupled, so the probe with the largest weight in one arbitrary fit is not the probe the model reliably wants.

**Conclusion: this model does not have a "most predictive gene."** At n=61 against 54,675 probes the selection is unstable enough that the question has no well-defined answer. Reporting any top-N list from this model would be reporting an artefact of one arbitrary split.

This is also the mechanism underneath the transfer failure. A signature that cannot survive a fold reshuffle *within* its own cohort was not going to survive an external cohort. Of the explanations offered below, this is the best-supported.

---

## Diagnosis: covariate shift between cohorts

The all-`Yes` collapse under the stored scaler is the second diagnostic result of the project.

Passing GSE23597 through the scaler fitted on GSE16879 gives **mean 6.566, sd 5.552** (training data through its own scaler gives 0 and 1 by construction). The decision function never drops below **+4.25** across all 45 patients (mean 38.3, max 67.5) — the boundary sits at 0, so the entire cohort is stranded on one side of it.

The two cohorts' expression distributions are offset by roughly 6.5 SD. Multiplied across 704 non-zero coefficients, a consistent per-probe offset sums into a decision function that never crosses zero; patient-specific biology never gets a vote. Re-centring GSE23597 on its own distribution restores mixed predictions — and **moves AUC by 0.015 (0.642 → 0.627)**. Centring relocates the threshold; it cannot create ranking signal that was not there.

That two radically different scaling regimes agree on AUC ≈ 0.63 is the strongest evidence that ~0.63 is what this model genuinely carries — and with a CI spanning 0.42–0.80, that is not distinguishable from chance.

**Candidate explanations for the transfer failure**, in descending order of support:

* **Dimensionality vs. sample size** — 54,675 features against 61 training patients. Directly evidenced by the selection-stability analysis above: the Lasso is selecting split-specific probes, not universal anti-TNF biology.
* **Covariate shift / batch effects** — demonstrated above, and partially corrected by per-cohort centring with no AUC benefit.
* **Phenotypic variance** — GSE16879 and GSE23597 may not define, score, or time "clinical response" identically (GSE23597 scores at week 8). Not separated by this analysis.

**On batch correction (ComBat):** deliberately not pursued. Per-cohort centring is a crude location-only batch correction; ComBat adds per-gene scale correction and empirical-Bayes shrinkage. It is a more sophisticated instrument aimed at a target that has already come back empty — and at n=45 with a CI touching 0.5, a marginal AUC improvement would not constitute a finding.

---

## Correction notice

Two headline claims in the previous version of this README were withdrawn after a leakage audit of this pipeline. Both are recorded here rather than quietly removed.

### 1. The "121 patients" were 121 samples — roughly 61 patients counted twice

GSE16879 contains **61 pre-treatment and 60 post-treatment biopsies**, and the original pipeline filtered out healthy controls but never filtered on timepoint — despite the external validation notebook correctly applying exactly this filter (`Time == 'W0'`) to GSE23597 for exactly this reason. The asymmetry between the two notebooks is what surfaced the bug.

Because response is a patient-level label, a random train/test split placed the same patient's pre- and post-treatment samples on both sides of the split with identical labels. With 54,675 features, the model did not need to learn anti-TNF biology — recognising a patient's transcriptomic fingerprint and reading the label off it is easier, and every extra feature sharpens the fingerprint.

The tell was in the tuning behaviour. On leaked data the inner loop chose `C = 100` and kept **10,457 genes** — walking to the *weakest* regularisation available, because more genes meant a better fingerprint. On the fixed, baseline-only data the same grid chose `C = 0.01` and kept **89 genes**: with the fingerprint gone, every extra gene is only an opportunity to fit noise, and the tuner clamps down. Same code, same grid, opposite ends of the search space.

**Affected claims:** the previously reported 72.00% internal accuracy, and the "80/20 partition … eliminating data leakage" description of the validation strategy. Neither survives.

### 2. The CLDN8 finding is retracted

The previous README reported probe `214598_at` → **CLDN8 (Claudin-8)** as the strongest positive predictor of response, and concluded that "the machine learning model independently discovered that baseline epithelial barrier robustness is a primary indicator for Infliximab therapeutic success."

That finding came from a model trained on duplicated patients at `C = 1.0` (an untuned default) retaining 1,245 genes. **Under leakage-free, baseline-only, per-fold-standardised nested CV, CLDN8 does not survive the Lasso penalty at all.** The claim is unsupported by this data and should not be cited.

The selection-stability analysis explains *why* it was never safe: CLDN8 was rank-1 in one fit, and rank-1 in one fit is not a finding at this sample size.

---

## What this project actually demonstrates

Not a working biomarker. What survives audit:

* A leakage bug found from an asymmetry between two of the project's own notebooks, confirmed empirically before being acted on, and fixed.
* An honest internal estimate (68.97% ± 9.13% vs a 54.1% baseline) that is robust to standardisation.
* A retracted headline biological finding, killed by the project's own audit rather than a reviewer's — and a stability analysis explaining why that class of finding was never safe here.
* A demonstrated ~6.5 SD covariate shift between cohorts, with the all-`Yes` collapse as direct evidence.
* A worked demonstration that accuracy is the wrong metric under prevalence shift, and why AUC is not.

The limiting factor is sample size: 61 training patients against 54,675 features, and 45 external patients producing an AUC CI half a unit wide. No amount of methodological care closes that gap — it is a data-collection problem, and naming it is more useful than another correction step.

---

## Repository structure

* `01_exploratory_analysis.ipynb` — initial exploration, debugging, experimental parsing for the discovery cohort.
* `02_final_pipeline.ipynb` — the discovery pipeline: baseline filtering, nested CV, final model fit, selection-stability analysis.
* `03_multi_cohort_evaluation_draft.ipynb` — scratchpad for longitudinal masking and string-engineering bugs on the second dataset.
* `04_multi_cohort_validation_final.ipynb` — external validation, both scaling regimes, AUC and bootstrap CI.
* `infliximab_coefficients.csv` — all 704 non-zero coefficients, sorted by absolute weight, with a `folds_selected` column (0–5) recording how many outer folds retained each probe.
* `infliximab_lasso_model.pkl` — the serialised pipeline (scaler + classifier).
* `.gitignore` — filters out local raw NCBI data folders.

---

## How to run this project locally

### 1. Prerequisites

Python 3.11+ with:

```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Order

Run `02_final_pipeline.ipynb` top to bottom first — it downloads GSE16879, fits the model, and writes `infliximab_lasso_model.pkl`. Then run `04_multi_cohort_validation_final.ipynb`, which downloads GSE23597 and loads that pickle.

**Restart Kernel → Run All, rather than running cells piecemeal.** Both notebooks are order-dependent: `02` writes the pickle `04` reads, and stale kernel state can silently score an old serialised model against fresh data.
