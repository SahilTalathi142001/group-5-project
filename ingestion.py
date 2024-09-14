import sys
from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Data Processing") \
    .getOrCreate()

# Define the source and target S3 paths
source_s3_path = "s3://datasource-bucket-dataops/used-cars-dataset/"
target_s3_path = "s3://datalake-bucket-dataops/used-cars-ingestion/"

# Read the data from the source S3 bucket (CSV format)
df = spark.read.option("header", "true").option("sep", ",").csv(source_s3_path)

# Coalesce the DataFrame to a single partition
df_single_file = df.coalesce(1)

# Write the DataFrame as a single Parquet file to the target S3 bucket
df_single_file.write.mode("overwrite").parquet(target_s3_path, compression="snappy")

# Stop the SparkSession
spark.stop()
