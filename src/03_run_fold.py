"""Resumable per-fold runner for the genomic value test. Usage: python 03_run_fold.py <fold_idx>"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines.utils import concordance_index
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("g", Path(__file__).parent/"03_genomic_value_test.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

fold_idx = int(sys.argv[1])
df = pd.read_csv(g.DATA, low_memory=False)
df = df[df["death_from_cancer"].notna()].copy()
df["event"] = (df["death_from_cancer"] != "Living").astype(int)
df["duration"] = df["overall_survival_months"].astype(float)
df = df[df["duration"].notna() & (df["duration"] >= 0)].reset_index(drop=True)
known = set(g.CLINICAL) | g.OUTCOME_DERIVED | {"patient_id"}
mrna = [c for c in df.columns if c not in known and not c.endswith("_mut")
        and pd.api.types.is_numeric_dtype(df[c])]
y = df["overall_survival"].astype(int).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr, te = list(skf.split(df, y))[fold_idx]
out = {"fold": fold_idx, "n_mrna": len(mrna)}
for k, fl in {"clinical": g.CLINICAL, "clinical_plus_mrna": g.CLINICAL + mrna}.items():
    pre = g.preprocessor(df, fl)
    Xtr = pre.fit_transform(df[fl].iloc[tr]); Xte = pre.transform(df[fl].iloc[te])
    Xtr = np.asarray(Xtr.todense()) if hasattr(Xtr, "todense") else Xtr
    Xte = np.asarray(Xte.todense()) if hasattr(Xte, "todense") else Xte
    gb = GradientBoostingClassifier(random_state=42).fit(Xtr, y[tr])
    auc = roc_auc_score(y[te], gb.predict_proba(Xte)[:, 1])
    ytr = Surv.from_arrays(df["event"].iloc[tr].astype(bool), df["duration"].iloc[tr])
    rsf = RandomSurvivalForest(n_estimators=100, max_features="sqrt",
                               min_samples_leaf=15, random_state=42, n_jobs=4).fit(Xtr, ytr)
    ci = concordance_index(df["duration"].iloc[te], -rsf.predict(Xte), df["event"].iloc[te])
    out[k] = {"auc": float(auc), "cindex": float(ci)}
    print(k, out[k], flush=True)
with open(g.OUT/"genomic_folds.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
