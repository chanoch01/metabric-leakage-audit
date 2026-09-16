"""
METABRIC leakage audit — Step 3: do genomic features add value over clinical
predictors once leakage is removed?

Design: paired comparison on identical stratified 5-fold splits (seed 42).
  clinical            : corrected clinical predictors (as Steps 1-2)
  clinical + mRNA     : corrected clinical + all mRNA z-score columns
Metrics: GB ROC-AUC (classification) and RSF C-index (survival).
Per-fold paired differences reported with mean and SD.
Results -> results/genomic_value_test.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines.utils import concordance_index

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "METABRIC_RNA_Mutation.csv"
OUT = ROOT / "results"

CLINICAL = [
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
OUTCOME_DERIVED = {"overall_survival", "overall_survival_months", "death_from_cancer", "event", "duration"}


def preprocessor(df, features):
    num = [c for c in features if df[c].dtype != object]
    cat = [c for c in features if df[c].dtype == object]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=10))]), cat),
    ])


def run():
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df["death_from_cancer"].notna()].copy()
    df["event"] = (df["death_from_cancer"] != "Living").astype(int)
    df["duration"] = df["overall_survival_months"].astype(float)
    df = df[df["duration"].notna() & (df["duration"] >= 0)].reset_index(drop=True)

    # mRNA z-score columns: numeric columns outside clinical/outcome sets
    known = set(CLINICAL) | OUTCOME_DERIVED | {"patient_id"}
    mrna = [c for c in df.columns
            if c not in known and not c.endswith("_mut")
            and pd.api.types.is_numeric_dtype(df[c])]
    feats = {"clinical": CLINICAL, "clinical_plus_mrna": CLINICAL + mrna}
    y = df["overall_survival"].astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(df, y))
    res = {"n": int(len(df)), "n_mrna_features": len(mrna), "auc": {}, "cindex": {}}

    per_fold = {k: {"auc": [], "cindex": []} for k in feats}
    for tr, te in folds:
        for k, fl in feats.items():
            pre = preprocessor(df, fl)
            Xtr = pre.fit_transform(df[fl].iloc[tr])
            Xte = pre.transform(df[fl].iloc[te])
            Xtr = np.asarray(Xtr.todense()) if hasattr(Xtr, "todense") else Xtr
            Xte = np.asarray(Xte.todense()) if hasattr(Xte, "todense") else Xte
            gb = GradientBoostingClassifier(random_state=SEED).fit(Xtr, y[tr])
            per_fold[k]["auc"].append(
                roc_auc_score(y[te], gb.predict_proba(Xte)[:, 1]))
            ytr = Surv.from_arrays(df["event"].iloc[tr].astype(bool),
                                   df["duration"].iloc[tr])
            rsf = RandomSurvivalForest(n_estimators=100, max_features="sqrt", min_samples_leaf=15,
                                       random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
            per_fold[k]["cindex"].append(
                concordance_index(df["duration"].iloc[te], -rsf.predict(Xte),
                                  df["event"].iloc[te]))

    for metric in ["auc", "cindex"]:
        a = np.array(per_fold["clinical"][metric])
        b = np.array(per_fold["clinical_plus_mrna"][metric])
        res[metric] = {
            "clinical_mean": float(a.mean()), "clinical_sd": float(a.std()),
            "clinical_plus_mrna_mean": float(b.mean()),
            "clinical_plus_mrna_sd": float(b.std()),
            "paired_diff_mean": float((b - a).mean()),
            "paired_diff_sd": float((b - a).std()),
            "per_fold_diff": [float(v) for v in (b - a)],
        }
        print(metric, res[metric])
    (OUT / "genomic_value_test.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    run()
