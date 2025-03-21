from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("RetailSalesForecasting").getOrCreate()

# Correct file path format
file_path = r"C:\Users\user\Desktop\Surya Project\Major Project\Scripts\monthwise_synthetic_retail_sales_data.csv"

# Debugging: Check if file exists
import os
print("File Exists:", os.path.exists(file_path))

# Load CSV
df = spark.read.csv(file_path, header=True, inferSchema=True)

df.show(650)
