import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, to_date, month
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import Row

# Explicitly set Python environment variables for PySpark
os.environ["PYSPARK_PYTHON"] = "C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python39\\python.exe"  # Replace with actual Python path
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python39\\python.exe"  # Replace with actual Python path

# Initialize Spark session with proper configurations
spark = SparkSession.builder \
    .appName("RetailSalesPrediction") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.python.worker.timeout", "300") \
    .config("spark.executor.heartbeatInterval", "10s") \
    .config("spark.network.timeout", "500s") \
    .getOrCreate()

# Load dataset
file_path = "monthwise_synthetic_retail_sales_data.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Validate dataset
print(f"Initial Dataset Row Count: {df.count()}")
df.show(5)

# Convert Date column to proper DateType and extract month
df = df.withColumn("Date", to_date(col("Date"), "dd-MM-yyyy"))
if df.filter(col("Date").isNull()).count() > 0:
    print("Warning: Null values found in 'Date' column after conversion!")

df = df.withColumn("Month", month(col("Date")))

# Clean Discount column (remove % and convert to double)
df = df.withColumn("Discount", regexp_replace(col("Discount"), "%", "").cast("double"))

# Fill missing values
df = df.fillna({"Month": 1, "Price": 0.0, "Quantity": 0, "Discount": 0.0, "Sales": 0.0})

# Validate missing values after filling
missing_value_counts = {col_name: df.filter(col(col_name).isNull()).count() for col_name in df.columns}
print("Missing values per column after filling: ", missing_value_counts)

# Feature engineering
assembler = VectorAssembler(inputCols=["Month", "Price", "Quantity", "Discount"], outputCol="features")
data = assembler.setHandleInvalid("skip").transform(df)

# Split data into train and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Handle empty training data
if train_data.count() == 0:
    print("Training dataset is empty. Using entire dataset for training...")
    train_data = data
    test_data = spark.createDataFrame([], data.schema)

# Train the model
lr = LinearRegression(featuresCol="features", labelCol="Sales", regParam=0.1, elasticNetParam=0.8)
model = lr.fit(train_data)

# Generate future dates dynamically (next 12 months)
latest_date = df.agg({"Date": "max"}).collect()[0][0]
if latest_date is None:
    raise ValueError("The 'Date' column is empty or contains invalid values!")

future_months = [(latest_date.replace(day=1).month + i - 1) % 12 + 1 for i in range(1, 13)]
future_data = spark.createDataFrame([Row(Month=m, Price=0.0, Quantity=0, Discount=0.0) for m in future_months])

# Transform future data
future_data = assembler.transform(future_data)

# Predict future sales
future_predictions = model.transform(future_data)
future_predictions.select("Month", "prediction").show()

print("Sales predictions for the next 12 months completed successfully.")
