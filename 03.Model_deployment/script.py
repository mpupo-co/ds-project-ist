import joblib
from data_pipeline import DataPipeline, load_dataset
TRAINING_DATASET = "data/traffic_accidents.csv"

if __name__ == "__main__":
    pipeline = DataPipeline(
        scaling="zscore",
        #balancing="smote",
        feature_selection="redundant",
    )
    training_df = load_dataset(TRAINING_DATASET)
    pipeline.fit(training_df)
    joblib.dump(pipeline, "models/pipeline.joblib")