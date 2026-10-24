import argparse
import os
from dateutil.relativedelta import relativedelta
import pyspark


from data_processing.bronze_table import process_bronze_table


def main(snapshotdate,
         csv_file_path,
          partition_prefix, 
          bronze_directory):
    print('\n\n---starting job---\n\n')
    print('ingesting snapshot date:', snapshotdate)
    print('csv file path:', csv_file_path)
    print('partition prefix:', partition_prefix)
    print('bronze directory:', bronze_directory)

    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)
    # run data processing
    process_bronze_table(
            csv_file_path=csv_file_path,
            partition_prefix=partition_prefix,
            snapshot_date_str=date_str,
            bronze_directory=bronze_directory,
            spark=spark
    )
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--csv_file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--partition_prefix", type=str, required=True, help="Prefix for the partition")
    parser.add_argument("--bronze_directory", type=str, required=True, help="Path to the bronze directory")

    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate,
         args.csv_file_path,
         args.partition_prefix,
         args.bronze_directory)
