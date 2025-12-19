from __future__ import annotations

from io import StringIO
import time
from typing import TYPE_CHECKING, Dict, List

import pandas as pd
from fastapi import HTTPException, UploadFile

if TYPE_CHECKING:
    from model_registry import ModelRegistry

def _log_stage(name: str, start_time: float) -> None:
    duration = time.perf_counter() - start_time
    print(f"[pipeline] {name} took {duration:.2f}s")

def single_prediction(file: UploadFile,
                      model_name: str | None,
                      registry: "ModelRegistry") -> Dict[str, object]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = file.file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content))

    if df.shape[0] != 1:
        raise HTTPException(status_code=400, detail="Only single record prediction is supported")

    return registry.predict(df, model_name)

def model_evaluation(file: UploadFile,
                     registry: "ModelRegistry",
                     target: str = "Cancelled") -> List[Dict[str, object]]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    t0 = time.perf_counter()
    content = file.file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content))
    _log_stage("Loading evaluation dataset", t0)
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in dataset")
    return registry.evaluate(df, target=target)