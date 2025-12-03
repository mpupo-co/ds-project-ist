#!/usr/bin/env python3

import os

files = [
    "DT_cf.py",
    "KNN_cf.py",
    "MLP_cf.py",
    "LR_cf.py",
    #"NB_cf.py",
    "RF_cf.py",
]

for f in files:
    print(f"\nRunning {f}...")
    os.system(f"python3 {f}")
