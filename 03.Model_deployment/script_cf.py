import joblib
from data_pipeline_cf import DataPipeline, load_dataset

if __name__ == "__main__":
    pipeline = DataPipeline(
        scaling="minmax",
        balancing="smote",
        feature_selection="redundant",
    )
    dataset = "data/Combined_Flights_2022.csv"
    df = load_dataset(dataset)
    df_sampled = df.sample(frac=0.01, replace=False, random_state=42)
    pipeline.fit(df_sampled)
    joblib.dump(pipeline, "models/pipeline_cf.joblib")