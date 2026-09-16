"""
METABRIC leakage audit — Step 2: censoring-aware survival modelling (corrected
predictors only) + calibration of the corrected classifier.

Event definition (overall survival): event = 1 if death_from_cancer is
'Died of Disease' or 'Died of Other Causes'; 0 (censored) if 'Living'.
Duration = overall_survival_months. The single record with missing
death_from_cancer is dropped.

Models: Cox PH (lifelines, penalised) and Random Survival Forest
(scikit-survival). Evaluation: Harrell's C-index via 5-fold CV with
preprocessing fitted inside folds. Calibration: 5-fold out-of-fold predicted
probabilities from the corrected GB classifier -> calibration table + Brier.
Results -> results/survival_analysis.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "METABRIC_RNA_Mutation.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

from importlib import import_module
step1 = import_module("01_classification_audit") if False else None  # avoid module-name issue
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
    df = df[df["duration"].notna() & (df["duration"] >= 0)]

    consistency = float(
        (df["event"] == (1 - df["overall_survival"])).mean()
    )  # expect ~1.0 if overall_survival: 1 = living

    X = df[CLINICAL_CORRECTED]
    results = {
        "n": int(len(df)),
        "events": int(df["event"].sum()),
        "event_vs_overall_survival_consistency": consistency,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cox_c, rsf_c = [], []
    for tr, te in kf.split(X):
        pre = preprocessor(df, CLINICAL_CORRECTED)
        Xtr = pre.fit_transform(X.iloc[tr])
        Xte = pre.transform(X.iloc[te])
        names = [f"x{i}" for i in range(Xtr.shape[1])]
        # Cox PH (L2-penalised for stability with one-hot dummies)
        trdf = pd.DataFrame(np.asarray(Xtr.todense()) if hasattr(Xtr, "todense") else Xtr,
                            columns=names)
        trdf["duration"] = df["duration"].iloc[tr].values
        trdf["event"] = df["event"].iloc[tr].values
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(trdf, duration_col="duration", event_col="event")
        tedf = pd.DataFrame(np.asarray(Xte.todense()) if hasattr(Xte, "todense") else Xte,
                            columns=names)
        risk = cph.predict_partial_hazard(tedf)
        cox_c.append(concordance_index(df["duration"].iloc[te], -risk,
                                       df["event"].iloc[te]))
        # RSF
        ytr = Surv.from_arrays(df["event"].iloc[tr].astype(bool),
                               df["duration"].iloc[tr])
        rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                                   random_state=SEED, n_jobs=-1)
        Xtr_d = np.asarray(Xtr.todense()) if hasattr(Xtr, "todense") else Xtr
        Xte_d = np.asarray(Xte.todense()) if hasattr(Xte, "todense") else Xte
        rsf.fit(Xtr_d, ytr)
        rrisk = rsf.predict(Xte_d)
        rsf_c.append(concordance_index(df["duration"].iloc[te], -rrisk,
                                       df["event"].iloc[te]))
    results["cox_cindex_mean"] = float(np.mean(cox_c))
    results["cox_cindex_sd"] = float(np.std(cox_c))
    results["rsf_cindex_mean"] = float(np.mean(rsf_c))
    results["rsf_cindex_sd"] = float(np.std(rsf_c))
    print("Cox C-index:", results["cox_cindex_mean"], "+/-", results["cox_cindex_sd"])
    print("RSF C-index:", results["rsf_cindex_mean"], "+/-", results["rsf_cindex_sd"])

    # --- calibration of corrected GB classifier (out-of-fold) ---------------
    y = df["overall_survival"].astype(int).values
    pipe = Pipeline([("pre", preprocessor(df, CLINICAL_CORRECTED)),
                     ("clf", GradientBoostingClassifier(random_state=SEED))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba",
                              n_jobs=-1)[:, 1]
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10)
    results["gb_calibration"] = {
        "mean_predicted": [float(v) for v in mean_pred],
        "fraction_positive": [float(v) for v in frac_pos],
        "brier": float(np.mean((proba - y) ** 2)),
    }
    (OUT / "survival_analysis.json").write_text(json.dumps(results, indent=2))
    print("calibration brier:", results["gb_calibration"]["brier"])


if __name__ == "__main__":
    run()
