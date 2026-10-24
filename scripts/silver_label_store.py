import os
import argparse
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, split
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from data_processing.silver_table import process_lms_silver_table

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
    silver_loan_daily_directory = "datamart/silver/loan_daily"

    if not os.path.exists(silver_loan_daily_directory):
        os.makedirs(silver_loan_daily_directory)

    process_lms_silver_table(
        snapshot_date_str=date_str,
        bronze_lms_directory="datamart/bronze/lms",
        silver_loan_daily_directory=silver_loan_daily_directory,
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
