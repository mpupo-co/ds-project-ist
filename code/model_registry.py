import json
from fastapi import HTTPException
import joblib
from pathlib import Path
from pandas import DataFrame
from typing import Any, Callable
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from helper import _log_stage
import time

RANDOM_STATE = 42
CLASS_EVAL_METRICS: dict[str, Callable] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "auc": roc_auc_score,
    "f1": f1_score,
}

class ModelRegistry:
    def __init__(self, descriptor_path: Path):
        self.descriptor_path = descriptor_path
        self.models: dict[str, Any] = {}
        self.hyperparams: dict[str, dict] = {}
        self.pipeline = None
        self.default_model: str | None = None
        self.test_size: float = 0.25

    def load(self) -> None:
        with open(self.descriptor_path) as f:
            descriptor = json.load(f)

        base_dir = self.descriptor_path.parent
        # Load pipeline
        print(f"Loading pipeline from {base_dir / descriptor['pipeline']}...")
        self.pipeline = joblib.load(base_dir / descriptor["pipeline"])
        # Load models
        for model_desc in descriptor["models"]:
            name = model_desc["name"]
            self.models[name] = joblib.load(base_dir / model_desc["file"])
            with open(base_dir / model_desc["hyperparams"]) as f:
                self.hyperparams[name] = json.load(f)

        self.default_model = descriptor["default_model"]

    def split_train_test_data(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        train = df.sample(frac=1-self.test_size, random_state=RANDOM_STATE)
        test = df.drop(train.index)
        return train, test    

    def list_models(self) -> list[str]:
        return list(self.models.keys())

    def predict(self, df: DataFrame, model_name: str | None = None) -> dict[str, Any]:
        try:
            model_name = model_name or self.default_model

            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            t = time.perf_counter()
            X = self.pipeline.transform(df, True)
            _log_stage("Transforming data for prediction", t)
            t = time.perf_counter()
            model = self.models[model_name]
            result = model.predict(X)
            _log_stage(f"Making prediction with model '{model_name}'", t)
            result = result[0]
            if hasattr(result, "item"):
                result = result.item()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")
        return {
            "model": model_name,
            "prediction": result,
        }

    def evaluate(self, df: DataFrame, target: str = "class") -> list[dict[str, Any]]:
        results = []
        try:
            t = time.perf_counter()
            df = self.pipeline.transform(df)
            _log_stage("Transforming data for evaluation", t)
            t = time.perf_counter()
            train, test = self.split_train_test_data(df)
            trnY = train.pop(target).values
            trnX = train.values
            tstY = test.pop(target).values
            tstX = test.values        
            for name, model in self.models.items():
                t = time.perf_counter()
                scores: dict = {}
                prd_trn = model.predict(trnX)
                prd_tst = model.predict(tstX)
                for key in CLASS_EVAL_METRICS:
                    scores[key] = [
                        CLASS_EVAL_METRICS[key](trnY, prd_trn),
                        CLASS_EVAL_METRICS[key](tstY, prd_tst),
                    ]
                results.append(
                    {
                        "model": name,
                        "scores": scores,
                        "hyperparams": self.hyperparams[name],
                    }
                )
                _log_stage(f"Making evaluation with model '{name}'", t)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during model evaluation: {str(e)}")

        return results



