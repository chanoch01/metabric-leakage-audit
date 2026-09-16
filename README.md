# METABRIC Survival Prediction: Leakage Audit and Reproducible Reconstruction

**Status:** published to this repository in September 2026. Not peer reviewed. This is not a published paper and not a preprint.

**Author:** Enoch Ewusi Hagan (FRSS)

## Purpose

This project reconstructs and audits an earlier breast cancer survival prediction analysis. It measures how much outcome-derived information and evaluation design changed the reported predictive performance, and it rebuilds the analysis with leakage removed and censoring handled properly.

## Background

My 2024 MSc dissertation (University of Stirling) used the METABRIC cohort to predict overall survival. As submitted, it reported ROC-AUC up to 0.93 and concluded that adding genomic data improved prediction. Its predictors included variables derived from the outcome itself, and it listed Cox models and random survival forests in its aims without implementing them.

**Examiner feedback on the dissertation identified the target leakage.** I did not identify it independently.

This repository is separate 2026 work. The reconstruction, the leakage experiments, the survival models, the genomic comparison and the calibration assessment were all built in 2026 and were not part of the submitted dissertation. The dissertation is the starting material, not a co-claim of these results.

## What this repository does

Every modelling comparison below uses the same preprocessing and models, changing only the predictor set.

- **Full leakage experiment (set A):** clinical predictors plus `overall_survival_months` and `death_from_cancer`, reconstructing the submitted configuration.
- **Partial leakage experiment (set B):** clinical predictors plus `overall_survival_months` only.
- **Corrected clinical-only analysis (set C):** every outcome-derived field removed.
- **Survival modelling:** L2-penalised Cox proportional hazards (lifelines) and a random survival forest (scikit-survival), evaluated with Harrell's C-index.
- **Calibration and evaluation:** out-of-fold Brier score and a ten-bin calibration table for the corrected classifier.
- **Genomic feature comparison:** corrected clinical predictors against clinical plus 489 mRNA z-score features, on identical paired folds.
- **Reproducibility:** pinned dependencies, fixed seeds and a single script that regenerates every result.

## Key findings

All figures come from the JSON files in `results/`, produced by the code in `src/`. They are rounded to three decimal places here.

| Analysis | Result |
|---|---|
| Patients / events (survival analysis) | 1,903 / 1,102 |
| Set A, full leak: gradient boosting ROC-AUC | 0.9995 (≈ 1.00) |
| Set B, partial leak: gradient boosting ROC-AUC | 0.862 |
| Set C, corrected: gradient boosting ROC-AUC | 0.762 |
| Cox proportional hazards C-index | 0.679 ± 0.011 |
| Random survival forest C-index | 0.694 ± 0.016 |
| Corrected classifier Brier score | 0.193 |
| Adding 489 mRNA features: change in ROC-AUC | −0.019 ± 0.015 |
| Adding 489 mRNA features: change in C-index | −0.011 ± 0.008 |

A note on patient counts: the classification stage uses 1,904 patients. The survival and genomic stages drop one patient whose vital status is missing, leaving 1,903.

**Genomic result, read carefully:** no improvement was demonstrated in this particular unregularised evaluation. The mRNA features were added without feature selection or regularisation, and the five paired folds were not given a formal significance test. This does not show that genomic data has no predictive value.

The full write-up, including the leakage mechanism and the limitations, is in [`FINDINGS.md`](FINDINGS.md).

## Important limitation

These results come from a reconstruction on a single public cohort under one 5-fold scheme. The work has not been published or peer reviewed and has not been independently validated. See the Limitations section of `FINDINGS.md` before citing any figure.

## Research questions

1. How far can outcome-derived predictors inflate the apparent discrimination of a clinical prediction model, compared with an otherwise identical pipeline?
2. What performance do censoring-aware survival models reach once leakage is removed?
3. Does adding high-dimensional mRNA expression improve a leakage-free clinical baseline under a paired comparison?
4. What evaluation practices (fold-contained preprocessing, paired comparisons, calibration) are needed before a reported performance figure can be trusted?

## Reproducibility

- **Python:** 3.11 or later (required by the pinned numpy 2.4.4). The exact interpreter version used for the stored results was not recorded.
- **Dependencies:** pinned in `requirements.txt` (scikit-learn 1.9.0, scikit-survival 0.28.0, lifelines 0.30.3, numpy 2.4.4, pandas 2.3.3).
- **Random seed:** 42 throughout (cross-validation splits and models).
- **Preprocessing:** median or most-frequent imputation, one-hot encoding and scaling, fitted inside each training fold only and applied to the held-out fold.
- **Cross-validation:** stratified 5-fold for classification and the genomic test; 5-fold for the survival models.

```bash
# 1. obtain the dataset and place it at data/METABRIC_RNA_Mutation.csv (see Dataset below)
# 2. install pinned dependencies
pip install -r requirements.txt
# 3. regenerate everything
bash run_all.sh
```

Outputs are written to `results/`: `classification_audit.json`, `survival_analysis.json`, `genomic_folds.jsonl` and `summary.json`. `src/04_aggregate_and_report.py` prints the headline table.

## Dataset

The raw METABRIC data is **not included** in this repository. See [`data/README.md`](data/README.md) for the expected file and its columns.

The file used is `METABRIC_RNA_Mutation.csv` (1,904 patients, clinical fields, mRNA z-scores and mutation status), commonly distributed on Kaggle as "Breast Cancer Gene Expression Profiles (METABRIC)". It derives from the METABRIC study on cBioPortal (https://www.cbioportal.org/study/summary?id=brca_metabric). Original studies: Curtis et al. (2012), *Nature*; Pereira et al. (2016), *Nature Communications*. Check the terms of whichever source you use before sharing the data.

## Repository structure

```
metabric-leakage-audit/
├── README.md
├── FINDINGS.md                      # full write-up and limitations
├── requirements.txt                 # pinned environment
├── run_all.sh                       # one-command reproduction
├── data/README.md                   # dataset source and columns (CSV not committed)
├── src/
│   ├── 01_classification_audit.py   # sets A, B, C under identical pipelines
│   ├── 02_survival_analysis.py      # Cox PH, random survival forest, calibration
│   ├── 03_genomic_value_test.py     # shared configuration for the genomic test
│   ├── 03_run_fold.py               # per-fold paired runner (clinical vs clinical + mRNA)
│   └── 04_aggregate_and_report.py   # builds results/summary.json and prints the table
└── results/                         # outputs used for every figure above
```

## Future work

- External validation on an independent cohort
- Regularised or dimension-reduced genomic models (for example elastic-net Cox)
- Stronger calibration analysis, including time-dependent calibration and integrated Brier score
- Competing-risks modelling where cause of death is relevant
- Uncertainty quantification for performance estimates (paired tests, bootstrap intervals, repeated cross-validation)
- Further evaluation such as time-dependent AUC and decision-curve analysis

## Changes made when publishing (September 2026)

No code logic, results or scientific conclusions were changed. The following was tidied before publication:

- README restructured, with the status line, research questions, Python version and future work added.
- README previously listed a `genomic_value_test.json` output that `run_all.sh` does not produce; that reference was removed.
- `run_all.sh` now clears `results/genomic_folds.jsonl` before the per-fold loop. The fold runner appends, so re-running without this would duplicate folds.
- One results heading in `FINDINGS.md` was reworded so it matches the qualified genomic conclusion already stated in its own text and limitations.
