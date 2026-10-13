
import pandas as pd

# Load the downloaded stock data
file_path = "data/stock_data.csv"

df = pd.read_csv(file_path, header=[0, 1], index_col=0)

# Display basic information
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== BASIC STATISTICS =====")
print(df.describe())
