import os
from pyspark.sql import SparkSession

# Set Python environment explicitly
os.environ["PYSPARK_PYTHON"] = "C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RetailSalesPrediction") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.python.worker.timeout", "300") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("Spark and Python are configured successfully!")
