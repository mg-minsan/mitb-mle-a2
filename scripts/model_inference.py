import os
import argparse
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark

def main(snapshot_date, model_name):    
    spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")


    config = {}
    config["snapshot_date_str"] = snapshot_date
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = model_name
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    print(config)
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)

    print("Model loaded successfully!" + config["model_artefact_filepath"])
    feature_location = "datamart/gold/features_store/"
    feature_store_file = f"gold_features_store_{config['snapshot_date'].strftime('%Y_%m_%d')}.parquet"
    feature_pdf = spark.read.option("header", "true").parquet(feature_location + feature_store_file).toPandas()
    print("extracted features for inference from:", feature_location + feature_store_file, "row count:", feature_pdf.shape[0])
    
    # --- extract features and prepare for inference ---
    # score only the columns the model was trained on, in the saved order
    X_inference = feature_pdf[model_artefact['feature_cols']]
    transformer = model_artefact['preprocessing_transformers']['transformer']
    X_inference = transformer.transform(X_inference)
    print("features for inference prepared successfully!")
    print("X_inference shape:", X_inference.shape)

    model = model_artefact["model"]
    y_inference = model.predict_proba(X_inference)[:, 1]
    y_inference_pdf = feature_pdf[["customer_id", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    print("inference results:", y_inference_pdf.shape[0])



    gold_inference_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
    print(gold_inference_directory)

    if not os.path.exists(gold_inference_directory):
        os.makedirs(gold_inference_directory)
        print("created directory:", gold_inference_directory)
    partition_name = config["model_name"][:-4] + "_" + config["snapshot_date"].strftime("%Y_%m_%d") + ".csv"
    filepath = os.path.join(gold_inference_directory, partition_name)
    y_inference_pdf.to_csv(filepath, index=False)
    
    print("inference results saved to:", filepath)
    spark.stop()
    print("\n\n----completed model inference for snapshot_date:", config["snapshot_date_str"], "----\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date in YYYY-MM-DD format")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., credit_model_2024_09_01.pkl)")
    args = parser.parse_args()
    main(args.snapshotdate, args.model)