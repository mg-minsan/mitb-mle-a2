import os
import argparse
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, split
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from data_processing.gold_table import process_features_gold_table 

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings

    spark.sparkContext.setLogLevel("ERROR")
    date_str = snapshotdate
    gold_feature_store_directory = "datamart/gold/features_store"
    
    if not os.path.exists(gold_feature_store_directory):
        os.makedirs(gold_feature_store_directory)

    process_features_gold_table(
        snapshot_date_str=date_str,
        silver_customers_directory="datamart/silver/customers",
        silver_financials_directory="datamart/silver/financials",
        silver_clickstream_directory="datamart/silver/clickstream",
        gold_features_store_directory=gold_feature_store_directory,
        spark=spark
    )
       
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n') 
    

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
