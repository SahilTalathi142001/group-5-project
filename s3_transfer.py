from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("S3 to S3 Transfer") \
    .getOrCreate()

# Define S3 paths
source_bucket = "s3://datasource-dataops-group5/vehicles/"
destination_bucket = "s3://datalake-dataops-group5/vehicles/"

# Read data from the source S3 bucket
df = spark.read.parquet(source_bucket)

# Write data to the destination S3 bucket
df.write.mode("overwrite").parquet(destination_bucket)

# Stop Spark session
spark.stop()
