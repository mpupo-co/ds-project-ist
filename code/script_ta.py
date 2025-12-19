import joblib
from data_pipeline_ta import DataPipeline, load_dataset

if __name__ == "__main__":
    pipeline = DataPipeline(
        scaling="minmax",
        feature_selection="redundant",
    )
    dataset = "data/traffic_accidents.csv"
    df = load_dataset(dataset)
    pipeline.fit(df)
    joblib.dump(pipeline, "models/pipeline_ta.joblib")