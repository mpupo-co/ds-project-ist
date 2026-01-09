#!/usr/bin/env python3

import os

files = [
    "ARIMA.py",
    "LSTMs.py",
    "mlp.py",
    "expo_soothing.py"
]

for f in files:
    print(f"\nRunning {f}...")
    os.system(f"python3 {f}")
