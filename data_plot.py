import pandas as pd
import matplotlib.pyplot as plt

# Correct file path for PyCharm
file_path = "C:/Users/user/Desktop/Surya Project/Major Project/monthwise_synthetic_retail_sales_data.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Ensure all relevant columns are numeric
numeric_cols = ["Sales", "Price", "Quantity"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Aggregate monthly sales
df_monthly = df.groupby(df["Date"].dt.to_period("M"))[numeric_cols].sum().reset_index()

# Convert 'Date' for plotting
df_monthly["Date"] = df_monthly["Date"].astype(str)

# Debugging: Check available columns
print("Columns after aggregation:", df_monthly.columns)

# Check if 'Sales' column exists
if "Sales" not in df_monthly.columns:
    print("Error: 'Sales' column not found. Available columns:", df_monthly.columns)
else:
    # Plot sales trend
    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly["Date"], df_monthly["Sales"], marker="o", linestyle="-", color="b")
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.title("Monthly Sales Trend")
    plt.grid(True)
    plt.show()
