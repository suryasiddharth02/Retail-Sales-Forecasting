from pyspark.sql import SparkSession
from pyspark.sql.functions import month, col, regexp_replace
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("RetailSalesForecasting").getOrCreate()

# Load dataset
file_path = "monthwise_synthetic_retail_sales_data.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Validate raw dataset
print(f"Raw Dataset Row Count: {df.count()}")
df.show(truncate=False)

# Convert Date column to Month
df = df.withColumn("Month", month(df["Date"]))

# Clean Discount column and convert to numeric
df = df.withColumn("Discount", regexp_replace(col("Discount"), "%", "").cast("double"))

# Handle null values
print(f"Rows Before Null Handling: {df.count()}")
df = df.fillna({"Month": 1, "Price": 0.0, "Quantity": 0, "Discount": 0.0, "Sales": 0.0})
print(f"Rows After Null Handling: {df.count()}")

# Assemble features
assembler = VectorAssembler(inputCols=['Month', 'Price', 'Quantity', 'Discount'], outputCol='features')
assembler.setHandleInvalid("skip")
data = assembler.transform(df)

# Validate feature assembly
print(f"Rows After Feature Assembly: {data.count()}")
data.select("features").show(truncate=False)

# Split data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Training Rows: {train_data.count()}, Testing Rows: {test_data.count()}")

# Handle empty training data
if train_data.count() == 0:
    print("Training dataset is empty. Using entire dataset for training...")
    train_data = data
    test_data = spark.createDataFrame([], data.schema)

# Train Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol='Sales')
model = lr.fit(train_data)

# Make predictions
if test_data.count() > 0:
    predictions = model.transform(test_data)
    predictions.show()

    # Evaluate the model
    rmse_evaluator = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="rmse")
    rmse = rmse_evaluator.evaluate(predictions)

    mae_evaluator = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="mae")
    mae = mae_evaluator.evaluate(predictions)

    r2_evaluator = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="r2")
    r2 = r2_evaluator.evaluate(predictions)

    # Calculate accuracy percentage
    sales_avg = df.select("Sales").na.drop().agg({"Sales": "avg"}).collect()[0][0]
    accuracy = (1 - (rmse / sales_avg)) * 100 if sales_avg != 0 else 0

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Prediction Accuracy: {accuracy:.2f}%")
else:
    print("Test dataset is empty. Skipping evaluation.")
