"""
METABRIC leakage audit — Step 1: classification pipelines, leaky vs corrected.

Pipelines compared (identical modelling, differing ONLY in predictor set):
  A_full_leak      : clinical + overall_survival_months + death_from_cancer dummies
                     (reconstructs the submitted dissertation's configuration)
  B_partial_leak   : clinical + overall_survival_months, no death_from_cancer
                     (state of the recovered Dissertation.ipynb, cell 18)
  C_corrected      : clinical only — all outcome-derived variables removed

Outcome: overall_survival (1 = living at last follow-up, 0 = died) — verified
against death_from_cancer before modelling.

Evaluation: stratified 5-fold CV, imputation/encoding/scaling fitted inside
each fold (no preprocessing leakage). Metrics: ROC-AUC, accuracy, Brier.
Seed fixed. Results written to results/classification_audit.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "METABRIC_RNA_Mutation.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

CLINICAL_CORRECTED = [
    "age_at_diagnosis", "type_of_breast_surgery", "cancer_type_detailed",
    "cellularity", "chemotherapy", "pam50_+_claudin-low_subtype", "cohort",
    "er_status_measured_by_ihc", "er_status", "neoplasm_histologic_grade",
    "her2_status_measured_by_snp6", "her2_status", "tumor_other_histologic_subtype",
    "hormone_therapy", "inferred_menopausal_state", "integrative_cluster",
    "primary_tumor_laterality", "lymph_nodes_examined_positive", "mutation_count",
    "nottingham_prognostic_index", "oncotree_code",
    "pr_status", "radio_therapy", "3-gene_classifier_subtype", "tumor_size",
    "tumor_stage",
]
LEAK_MONTHS = ["overall_survival_months"]
LEAK_VITAL = ["death_from_cancer"]


def build_pipeline(df: pd.DataFrame, features: list, model) -> Pipeline:
    num = [c for c in features if df[c].dtype != object]
    cat = [c for c in features if df[c].dtype == object]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def run():
    df = pd.read_csv(DATA, low_memory=False)
    # --- outcome sanity check ------------------------------------------------
    tab = pd.crosstab(df["overall_survival"], df["death_from_cancer"], dropna=False)
    living_share_when_1 = (
        df.loc[df.overall_survival == 1, "death_from_cancer"].eq("Living").mean()
    )
    outcome_check = {
        "crosstab": tab.to_dict(),
        "share_Living_when_overall_survival_1": float(living_share_when_1),
    }
    y = df["overall_survival"].astype(int)

    models = {
        "logreg": LogisticRegression(max_iter=2000, random_state=SEED),
        "rf": RandomForestClassifier(random_state=SEED),
        "gb": GradientBoostingClassifier(random_state=SEED),
    }
    variants = {
        "A_full_leak": CLINICAL_CORRECTED + LEAK_MONTHS + LEAK_VITAL,
        "B_partial_leak": CLINICAL_CORRECTED + LEAK_MONTHS,
        "C_corrected": CLINICAL_CORRECTED,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {"outcome_check": outcome_check, "n": int(len(df)), "cv": "stratified 5-fold, seed 42"}
    for vname, feats in variants.items():
        results[vname] = {}
        X = df[feats]
        for mname, model in models.items():
            pipe = build_pipeline(df, feats, model)
            cvres = cross_validate(
                pipe, X, y, cv=cv,
                scoring={"auc": "roc_auc", "acc": "accuracy", "brier": "neg_brier_score"},
                n_jobs=-1,
            )
            results[vname][mname] = {
                "auc_mean": float(np.mean(cvres["test_auc"])),
                "auc_sd": float(np.std(cvres["test_auc"])),
                "acc_mean": float(np.mean(cvres["test_acc"])),
                "brier_mean": float(-np.mean(cvres["test_brier"])),
            }
            print(vname, mname, results[vname][mname])
    (OUT / "classification_audit.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
