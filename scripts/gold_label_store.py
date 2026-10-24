import os
import argparse
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, split
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from data_processing.gold_table import process_labels_gold_table

def main(snapshotdate, dpd, mob):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings

    spark.sparkContext.setLogLevel("ERROR")
    date_str = snapshotdate
    gold_label_store_directory = "datamart/gold/label_store"
    
    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)

    process_labels_gold_table(
        snapshot_date_str=date_str,
        silver_loan_daily_directory="datamart/silver/loan_daily",
        gold_label_store_directory=gold_label_store_directory,
        dpd=dpd,
        mob=mob,
        spark=spark
    )
       
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n') 
    

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--dpd", type=int, required=True, help="Days past due threshold for labeling")
    parser.add_argument("--mob", type=int, required=True, help="Months on book threshold for labeling")

    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.dpd, args.mob)
