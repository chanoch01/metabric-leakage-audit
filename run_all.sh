#!/usr/bin/env bash
# Reproduce the full METABRIC leakage audit end-to-end.
# Requires: data/METABRIC_RNA_Mutation.csv  (see data/README.md for source)
set -euo pipefail
python3 src/01_classification_audit.py
python3 src/02_survival_analysis.py
rm -f results/genomic_folds.jsonl  # fold runner appends; start clean
for f in 0 1 2 3 4; do python3 src/03_run_fold.py "$f"; done
python3 src/04_aggregate_and_report.py
echo "Done. See results/ and FINDINGS.md"
