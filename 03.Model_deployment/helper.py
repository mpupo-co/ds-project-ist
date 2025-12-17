from fastapi import UploadFile, HTTPException
import pandas as pd
from io import StringIO
import time

def _log_stage(name, start_time):
    duration = time.perf_counter() - start_time
    print(f"[pipeline] {name} took {duration:.2f}s")

def single_prediction(file: UploadFile,
                      model_name: str | None,
                      registry) -> dict:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = file.file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content))

    if df.shape[0] != 1:
        raise HTTPException(status_code=400, detail="Only single record prediction is supported")

    result = registry.predict(df, model_name)
    return result

def model_evaluation(file: UploadFile,
                     registry) -> list[dict]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    t0 = time.perf_counter()
    content = file.file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content))
    _log_stage("Loading evaluation dataset", t0)
    target = "crash_type"
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in dataset")
    results = registry.evaluate(df, target=target)
    return results