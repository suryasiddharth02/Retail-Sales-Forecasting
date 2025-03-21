from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year
from pyspark.ml.feature import VectorAssembler, StringIndexer

# Initialize Spark session
spark = SparkSession.builder.appName("TrainModel").getOrCreate()

# Load your dataset
df = spark.read.csv("monthwise_synthetic_retail_sales_data.csv", header=True, inferSchema=True)

# Extract Year from Date if needed
if "Date" in df.columns:
    df = df.withColumn("Year", year(col("Date")))

# Convert categorical columns (Store, Region, Discount) into numerical indexes
indexers = ["Store", "Region", "Discount"]
for col_name in indexers:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_Index", handleInvalid="keep")
    df = indexer.fit(df).transform(df)

# Handle missing values
df = df.na.fill(0)  # Replace nulls with 0

# Define feature columns (remove Year if not available)
feature_columns = ["Sales", "Price", "Quantity", "Year", "Store_Index", "Region_Index", "Discount_Index"]
feature_columns = [col for col in feature_columns if col in df.columns]  # Keep only existing columns

# Create Vector Assembler
vector_assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="keep"
)

# Transform dataset
transformed_df = vector_assembler.transform(df)

# Show result
transformed_df.select("features").show(5, truncate=False)
