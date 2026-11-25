#!/usr/bin/env python3
"""Verify the model inputs (sequences_30s.npz) against the dataset. Exit code 1 on failure.

  python check_inputs.py
"""
import argparse
import sys
from pathlib import Path

from dataset.verify import verify

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequences", default=str(HERE / "data" / "dataset" / "sequences_30s.npz"))
    p.add_argument("--dataset", default=str(HERE / "data" / "dataset" / "dataset_30s.csv"))
    p.add_argument("--samples", type=int, default=300, help="random samples to check token-by-token")
    a = p.parse_args()
    print(f"Checking {a.sequences}")
    report = verify(a.sequences, a.dataset, n_samples=a.samples)
    report.print()
    print("RESULT:", "FAIL" if report.failed else "PASS")
    sys.exit(1 if report.failed else 0)
