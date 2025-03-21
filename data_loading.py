from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("RetailSalesForecasting").getOrCreate()

# Load CSV file
file_path = "monthwise_synthetic_retail_sales_data.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Show schema and first few rows
df.printSchema()
df.show(5)
