"""
METABRIC leakage audit — Step 4: aggregate all results into a single
machine-readable summary and a human-readable FINDINGS table.
Reads results/*.json and results/genomic_folds.jsonl; writes results/summary.json
and prints the headline table.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def main():
    clf = json.loads((OUT / "classification_audit.json").read_text())
    surv = json.loads((OUT / "survival_analysis.json").read_text())
    folds = [json.loads(l) for l in (OUT / "genomic_folds.jsonl").read_text().splitlines()]

    gv = {}
    for metric in ["auc", "cindex"]:
        a = np.array([r["clinical"][metric] for r in folds])
        b = np.array([r["clinical_plus_mrna"][metric] for r in folds])
        gv[metric] = {
            "clinical_mean": float(a.mean()), "clinical_sd": float(a.std()),
            "plus_mrna_mean": float(b.mean()), "plus_mrna_sd": float(b.std()),
            "paired_diff_mean": float((b - a).mean()),
            "paired_diff_sd": float((b - a).std()),
        }

    summary = {
        "outcome_integrity": {
            "share_Living_when_overall_survival_1":
                clf["outcome_check"]["share_Living_when_overall_survival_1"],
            "event_vs_overall_survival_consistency":
                surv["event_vs_overall_survival_consistency"],
            "n": surv["n"], "events": surv["events"],
        },
        "classification_auc_gb": {
            "A_full_leak": clf["A_full_leak"]["gb"]["auc_mean"],
            "B_partial_leak": clf["B_partial_leak"]["gb"]["auc_mean"],
            "C_corrected": clf["C_corrected"]["gb"]["auc_mean"],
        },
        "survival_cindex": {
            "cox": surv["cox_cindex_mean"], "rsf": surv["rsf_cindex_mean"],
        },
        "genomic_value": gv,
        "calibration_brier_gb_corrected": surv["gb_calibration"]["brier"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nHEADLINE RESULTS")
    print("-" * 60)
    print("Outcome integrity check:")
    print(f"  overall_survival==1 <=> death_from_cancer=='Living': "
          f"{summary['outcome_integrity']['share_Living_when_overall_survival_1']:.3f}")
    print(f"  n={summary['outcome_integrity']['n']}  "
          f"events={summary['outcome_integrity']['events']}")
    print("\nClassification GB ROC-AUC (5-fold CV):")
    print(f"  A full leak (months + vital status): "
          f"{summary['classification_auc_gb']['A_full_leak']:.4f}  (~1.00)")
    print(f"  B partial leak (months only)       : "
          f"{summary['classification_auc_gb']['B_partial_leak']:.3f}")
    print(f"  C corrected (clinical only)        : "
          f"{summary['classification_auc_gb']['C_corrected']:.3f}")
    print("\nCorrected survival models (C-index):")
    print(f"  Cox PH: {summary['survival_cindex']['cox']:.3f}   "
          f"RSF: {summary['survival_cindex']['rsf']:.3f}")
    print("\nGenomic added value (paired, corrected features):")
    print(f"  AUC   clinical {gv['auc']['clinical_mean']:.3f} -> "
          f"+mRNA {gv['auc']['plus_mrna_mean']:.3f}  "
          f"(diff {gv['auc']['paired_diff_mean']:+.3f})")
    print(f"  Cidx  clinical {gv['cindex']['clinical_mean']:.3f} -> "
          f"+mRNA {gv['cindex']['plus_mrna_mean']:.3f}  "
          f"(diff {gv['cindex']['paired_diff_mean']:+.3f})")


if __name__ == "__main__":
    main()
