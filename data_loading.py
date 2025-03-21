import pandas as pd

# Load the dataset
file_path = "monthwise_synthetic_retail_sales_data.csv"
df = pd.read_csv(file_path)

# Display basic information and first few rows
df.info(), df.head()
