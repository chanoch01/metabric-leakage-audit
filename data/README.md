# Dataset

**File expected here:** `METABRIC_RNA_Mutation.csv` (1,904 rows × 693 columns).

This is the widely circulated METABRIC clinical + mRNA z-score + mutation export
(the same file used by the original dissertation, loaded in the recovered notebook
from `/content/drive/MyDrive/Dissertation/METABRIC_RNA_Mutation.csv`). It is
commonly distributed via Kaggle ("Breast Cancer Gene Expression Profiles
(METABRIC)") and derives from cBioPortal's METABRIC study
(Curtis et al., 2012, *Nature*; Pereira et al., 2016, *Nature Communications*).

The CSV is deliberately not included. Check the redistribution terms of your
source before sharing it. Download it from your source and place it at
`data/METABRIC_RNA_Mutation.csv`.

## Columns used

- Outcome: `overall_survival` (1 = living at last follow-up, 0 = died).
- Excluded as outcome-derived (leakage): `overall_survival_months`,
  `death_from_cancer`.
- Corrected clinical predictors: 26 fields (demographics, tumour characteristics,
  receptor status, treatment, NPI, stage, grade, integrative cluster and so on); see
  `CLINICAL_CORRECTED` in `src/01_classification_audit.py`.
- Genomic-value test adds all mRNA z-score columns (~490 numeric gene columns,
  excluding `*_mut` mutation-status columns).

## Integrity check baked into the code

`src/01` and `src/02` verify that `overall_survival == 1` corresponds exactly to
`death_from_cancer == "Living"` (observed share 1.000), documenting precisely why
those fields constitute leakage.
