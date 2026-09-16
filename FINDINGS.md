# Target Leakage in METABRIC Breast Cancer Survival Prediction: Reconstruction and Corrected Analysis

Enoch Ewusi Hagan, FRSS. Analysis carried out in 2026 and published to this repository in September 2026. Not peer reviewed.

## Summary

My MSc dissertation (University of Stirling, 2024) predicted overall survival in breast cancer patients from the METABRIC cohort. It reported ROC-AUC up to 0.93 and concluded that adding genomic data significantly improved prediction. My examiner pointed out a serious flaw: some predictors had been derived from the outcome itself. In this work I rebuilt the pipeline from the original code, removed the leakage, fitted the survival models the dissertation had proposed but not implemented, and re-tested the claim about genomic data.

With the outcome-derived variables removed, gradient boosting reached a ROC-AUC of about 0.76. Cox and random survival forest models reached C-indices of about 0.68 and 0.69. Adding around 490 mRNA expression features to the corrected clinical model did not improve either metric in this evaluation, and both were slightly lower. The dissertation's main conclusion does not hold under this corrected, unregularised comparison.

## What was done in 2024 and what was done in 2026

**The 2024 dissertation, as submitted.** It applied random forests, gradient boosting, support vector machines, logistic regression and neural networks to METABRIC, reported ROC-AUC up to 0.93, and concluded that multi-omics integration improved performance. Its feature set included outcome-derived variables. Its aims mentioned Cox models, random survival forests and the concordance index, but it contained no survival model results. The target leakage was identified in examiner feedback; I did not find it myself.

**This repository (2026).** I rebuilt the pipeline from the original code, measured the effect of the leakage, fitted the survival models, tested the genomic claim on paired folds and checked calibration. None of the survival analysis, genomic comparison or calibration work existed in the submitted dissertation or in the recovered notebook.

The work has not been peer reviewed and is not a preprint. Every figure below is produced by the code in `src/`.

## The data and how the leakage happened

The METABRIC file has 1,904 patients. It includes a binary outcome, `overall_survival` (1 = alive at last follow-up, 0 = died), a follow-up time, `overall_survival_months`, and a vital status field, `death_from_cancer`, with the values "Living", "Died of Disease" and "Died of Other Causes". The recovered notebook used `overall_survival_months` as a predictor (feature list at cell 18), and an earlier feature importance analysis also included `death_from_cancer`, which came out as the most important predictor.

Both variables are functions of the outcome:

- `death_from_cancer == "Living"` means the same thing as `overall_survival == 1`. In this data they match exactly (the share of "Living" among survivors is 1.000), so including it more or less gives the model the answer.
- `overall_survival_months` is follow-up time. Patients who died and patients who survived have systematically different follow-up, so this variable carries a lot of information about the label.

Using either as a predictor raises apparent performance without adding any real prognostic information. That is target leakage.

## Method

Classification models were evaluated with stratified 5-fold cross-validation and survival models with standard 5-fold cross-validation, all with a fixed seed (42). Imputation (median or most frequent), one-hot encoding and scaling were fitted on each training fold only and then applied to the held-out fold, so no information passed between training and test data. Three predictor sets were compared using otherwise identical models:

- Set A, full leak: clinical predictors plus `overall_survival_months` and `death_from_cancer`, reconstructing the submitted configuration.
- Set B, partial leak: clinical predictors plus `overall_survival_months`, matching the recovered notebook.
- Set C, corrected: clinical predictors only, with every outcome-derived field removed.

The survival models (an L2-penalised Cox model and a random survival forest) used death from any cause as the event and `overall_survival_months` as the duration, and were evaluated with Harrell's concordance index. The genomic test compared the corrected clinical predictors with the same predictors plus all mRNA z-score columns, on identical paired folds, using gradient boosting (AUC) and a random survival forest (C-index). The code is in `src/`.

## Results

### 1. Leakage accounts for the high reported performance

Gradient boosting, 5-fold cross-validated ROC-AUC:

| Predictor set | ROC-AUC |
|---|---|
| A, full leak (follow-up time and vital status) | 0.9995 (about 1.00) |
| B, partial leak (follow-up time only) | 0.862 |
| C, corrected (clinical only) | 0.762 |

With both leaked variables included, the task becomes almost trivial. Follow-up time on its own still raises the AUC to 0.86, while the clinical-only model reaches 0.76. The 0.93 reported in the dissertation falls between the two leaked configurations, which is what you would expect from a leaked pipeline.

### 2. The corrected survival models perform moderately

On 1,903 patients with 1,102 events, the Cox model reached a C-index of 0.679 ± 0.011 and the random survival forest 0.694 ± 0.016. These are reasonable values for breast cancer survival models based on clinical variables, and they are consistent with the classification result.

### 3. No improvement from genomic features was demonstrated

Paired 5-fold comparison with corrected features:

| Metric | Clinical only | Clinical + mRNA (489 features) | Paired difference |
|---|---|---|---|
| Gradient boosting ROC-AUC | 0.763 ± 0.019 | 0.743 ± 0.011 | −0.019 ± 0.015 |
| Random survival forest C-index | 0.690 ± 0.017 | 0.679 ± 0.013 | −0.011 ± 0.008 |

Adding the mRNA features lowered discrimination slightly in almost every fold, which is the opposite of the dissertation's conclusion. The difference is small, and it is summarised over five paired folds without a formal significance test, so it should be read as a lack of demonstrated improvement and not as evidence that genomic data have no predictive value. This pattern is common when several hundred noisy features are added to a strong low-dimensional clinical signal without dimension reduction or regularised feature selection.

### 4. Calibration

The corrected gradient boosting classifier has an out-of-fold Brier score of 0.193. A ten-bin calibration table is saved in `results/survival_analysis.json` for later recalibration work.

## Interpretation

The dissertation's main result came from target leakage and says nothing reliable about the value of genomic data. The corrected analysis points to three lessons that apply beyond this dataset:

1. Outcome-derived predictors can produce almost perfect apparent performance (AUC about 1.00 here), so clinical prediction studies should check for leakage as a matter of routine.
2. The gap between reported and corrected performance can be large and can be measured directly (0.93 against 0.76 here).
3. A new data source should be shown to help against a leakage-free baseline. Here, the genomic features did not, although no feature selection or regularisation was applied (see Limitations).

## Limitations and next steps

**No formal test of the genomic comparison.** The difference between the clinical and clinical plus mRNA models is reported as a mean and standard deviation over five paired folds. I did not run a paired significance test or compute a bootstrap confidence interval, so the correct reading is that no improvement was demonstrated in this evaluation. Adding such a test is the first piece of further work.

**No genomic feature selection or regularisation.** Around 490 mRNA features were added directly to a strong clinical signal. A penalised model, such as an elastic-net Cox model, or a dimension-reduced representation might recover some value, and should be tried before drawing any wider conclusion about genomic data.

**The survival analysis is new work.** The recovered notebook had no survival modelling. The Cox and random survival forest analyses were built to match what the dissertation said it would do; they are not a re-run of earlier code.

**Limited evaluation.** Only discrimination and a single Brier score are reported. Time-dependent AUC, the integrated Brier score, competing risks and decision curve analysis would all make the survival evaluation stronger.

**One cohort and one split scheme.** All results come from METABRIC under a single 5-fold scheme with a fixed seed. Repeated cross-validation would give tighter variance estimates, and validation on an independent cohort (for example SEER or TCGA-BRCA) is the obvious next step.

**What the limitations do not change.** The leakage is present in the original work and was identified in examiner feedback, and its effect can be measured directly by comparing otherwise identical pipelines.

## Reproducibility

`bash run_all.sh` regenerates every number above from `data/METABRIC_RNA_Mutation.csv`. Package versions are pinned in `requirements.txt` (scikit-learn 1.9.0, scikit-survival 0.28.0, lifelines 0.30.3). All figures in this document come from the code in `src/`.
