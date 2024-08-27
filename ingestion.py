import sys
from pyspark.sql import SparkSession

try:
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("IngestionJob") \
        .getOrCreate()

    # Initialize the Glue job context if running in AWS Glue
    if 'JOB_NAME' in sys.argv:
        from awsglue.utils import getResolvedOptions
        from awsglue.context import GlueContext
        from awsglue.job import Job

        # Ensure GlueContext is initialized
        if 'glueContext' not in locals():
            glueContext = GlueContext(spark.sparkContext)
            spark = glueContext.spark_session

        # Initialize the Glue job
        args = getResolvedOptions(sys.argv, ['JOB_NAME'])
        job = Job(glueContext)
        job.init(args['JOB_NAME'], args)

    # Define the source and target S3 paths
    source_s3_path = "s3://datasource-dbda-group5/vehicles-dataset/"
    target_s3_path = "s3://datalake-dbda-group5/vehicles-ingestion/"

    # Read the data from the source S3 bucket (CSV format)
    df = spark.read \
        .option("header", "true") \
        .option("sep", ",") \
        .csv(source_s3_path)

    # Coalesce the DataFrame to a single partition
    df_single_file = df.coalesce(1)

    # Write the DataFrame as a single Parquet file to the target S3 bucket
    df_single_file.write \
        .mode("overwrite") \
        .parquet(target_s3_path, compression="snappy")

    # Commit the Glue job if running in AWS Glue
    if 'JOB_NAME' in sys.argv:
        job.commit()

except Exception as e:
    print(f"Error: {e}")

finally:
    # Stop the Spark Session
    spark.stop()
