from pyspark.sql import SparkSession
from pyspark.sql.functions import lag
from pyspark.sql.window import Window

# Step 1: Initialize Spark Session
spark = SparkSession.builder.appName("RetailSalesForecasting").getOrCreate()

# Step 2: Load Data
file_path = r"C:\Users\user\Desktop\Surya Project\Major Project\Scripts\monthwise_synthetic_retail_sales_data.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Step 3: Print Schema to Verify Columns
df.printSchema()
df.show(5)

# Step 4: Define Window Specification with the Correct Column Name
window_spec = Window().partitionBy("Store").orderBy("Date")  # Replace 'Store_ID' with 'Store'

# Step 5: Apply Lag Function for Previous Sales
df = df.withColumn("Previous_Sales", lag("Sales", 1).over(window_spec))

# Step 6: Show Results
df.show(10)
