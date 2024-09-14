import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType, LongType, DoubleType, TimestampType

# Initialize Spark Session
spark = SparkSession.builder.appName("Used Cars Data Transformation").getOrCreate()

# Input and Output S3 paths
s3_input_path = "s3://datalake-bucket-dataops/used-cars-ingestion/"
s3_output_path = "s3://datawarehouse-bucket-dataops/used-cars-transform/"

# Load the data from the S3 input path
used_cars = spark.read.option("header", "true").parquet(s3_input_path)

# Print schema and sample data for debugging
print("Schema before processing:")
used_cars.printSchema()
print("Sample data before processing:")
used_cars.show(5)

# Dropping the 'id' and 'county' columns
used_cars = used_cars.drop('id', 'county')

# Define boundaries for US lat/long
min_latitude = 24.51
max_latitude = 49.38
min_longitude = -171.83
max_longitude = -66.95

# Filtering for latitude and longitude within the US
used_cars = used_cars.filter(
    (F.col('lat').between(min_latitude, max_latitude)) &
    (F.col('long').between(min_longitude, max_longitude)) &
    (F.col('long') < 0) &
    (F.col('lat') > 0)
)

# Function to impute missing categorical values randomly
def random_imputer(df, column):
    non_missing_categories = df.select(column).where(F.col(column).isNotNull()).distinct().rdd.flatMap(lambda x: x).collect()
    if not non_missing_categories:
        print(f"No non-missing categories found for {column}. Skipping imputation.")
        return df
    random_values = 'array({})'.format(','.join(["'{}'".format(val) for val in non_missing_categories]))
    return df.withColumn(
        column, 
        F.when(F.col(column).isNull(), F.expr("element_at({}, cast(rand() * size({}) + 1 as int))".format(random_values, random_values))).otherwise(F.col(column))
    )

# Impute missing categorical values for paint_color and size
used_cars = random_imputer(used_cars, 'paint_color')
used_cars = random_imputer(used_cars, 'size')

# Fill missing 'VIN' with 'unknown'
used_cars = used_cars.fillna({'VIN': 'unknown'})

# Fill missing 'type', 'transmission', 'title_status', 'year', and 'fuel' with mode
for column in ['type', 'transmission', 'title_status', 'year', 'fuel']:
    mode_value = used_cars.groupBy(column).count().orderBy(F.col("count").desc()).first()
    if mode_value and mode_value[0] is not None:
        used_cars = used_cars.fillna({column: mode_value[0]})

# Convert year to integer and then to string
used_cars = used_cars.withColumn('year', F.col('year').cast(IntegerType()))
used_cars = used_cars.withColumn('year', F.col('year').cast(StringType()))

# Function to impute missing values based on most frequent values
def random_imputer2(df, column):
    non_missing_categories = df.groupBy(column).count().orderBy(F.col("count").desc()).limit(5).select(column).rdd.flatMap(lambda x: x).collect()
    if not non_missing_categories:
        print(f"No non-missing categories found for {column}. Skipping imputation.")
        return df
    random_values = 'array({})'.format(','.join(["'{}'".format(val) for val in non_missing_categories]))
    return df.withColumn(
        column, 
        F.when(F.col(column).isNull(), F.expr("element_at({}, cast(rand() * size({}) + 1 as int))".format(random_values, random_values))).otherwise(F.col(column))
    )

# Impute missing 'manufacturer' using the top 5 manufacturers
used_cars = random_imputer2(used_cars, 'manufacturer')

# Fill model column based on manufacturer and year
fill = F.concat(F.col('manufacturer'), F.lit(' '), F.col('year'))
used_cars = used_cars.withColumn('model', F.when(F.col('model').isNull(), fill).otherwise(F.col('model')))

# Filter out invalid odometer values and fill missing odometer with the mean
used_cars = used_cars.filter(F.col('odometer') > 20.0)
mean_odometer = used_cars.agg(F.avg('odometer')).first()
if mean_odometer and mean_odometer[0] is not None:
    used_cars = used_cars.fillna({'odometer': mean_odometer[0]})

# Fill missing description with 'no description'
used_cars = used_cars.fillna({'description': 'no description'})

# Replace '4wd' with 'fwd' in drive column
used_cars = used_cars.withColumn('drive', F.when(F.col('drive') == '4wd', 'fwd').otherwise(F.col('drive')))

# Remove rows with zero price
used_cars = used_cars.filter(F.col('price') != 0)

# Coalesce to reduce small partitions before writing
used_cars = used_cars.coalesce(1)

# Change the data types of columns
used_cars = used_cars.withColumn('price', F.col('price').cast(LongType())) \
                     .withColumn('year', F.col('year').cast(IntegerType())) \
                     .withColumn('odometer', F.col('odometer').cast(LongType())) \
                     .withColumn('lat', F.col('lat').cast(DoubleType())) \
                     .withColumn('long', F.col('long').cast(DoubleType())) \
                     .withColumn('posting_date', F.col('posting_date').cast(TimestampType())) \
                     .withColumn('manufacturer', F.col('manufacturer').cast(StringType())) \
                     .withColumn('model', F.col('model').cast(StringType())) \
                     .withColumn('condition', F.col('condition').cast(StringType())) \
                     .withColumn('cylinders', F.col('cylinders').cast(StringType())) \
                     .withColumn('fuel', F.col('fuel').cast(StringType())) \
                     .withColumn('title_status', F.col('title_status').cast(StringType())) \
                     .withColumn('transmission', F.col('transmission').cast(StringType())) \
                     .withColumn('VIN', F.col('VIN').cast(StringType())) \
                     .withColumn('drive', F.col('drive').cast(StringType())) \
                     .withColumn('size', F.col('size').cast(StringType())) \
                     .withColumn('type', F.col('type').cast(StringType())) \
                     .withColumn('paint_color', F.col('paint_color').cast(StringType()))

# Write the final DataFrame to the specified S3 output path in Parquet format
used_cars.write.mode('overwrite').parquet(s3_output_path)

# Stop the Spark session
spark.stop()
