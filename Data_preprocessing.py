import pandas as pd  # Import Pandas

# Load the dataset
file_path = "C:\\Users\\user\\Desktop\\Surya Project\\Major Project\\monthwise_synthetic_retail_sales_data.csv"  # Update file path
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Convert 'Discount' from percentage string to float
df["Discount"] = df["Discount"].str.replace("%", "").astype(float)

# Display updated data types and missing values
print("Updated Data Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Save the cleaned data (optional)
df.to_csv("C:\\Users\\user\\Desktop\\Surya Project\\Major Project\\cleaned_data.csv", index=False)
