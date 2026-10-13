
import pandas as pd

# Load historical stock data
df = pd.read_csv(
    "data/stock_data.csv",
    header=[0, 1],
    index_col=0
)

# Extract the stock price columns
data = pd.DataFrame(index=df.index)

data["Open"] = df["Open"]["RELIANCE.NS"]
data["High"] = df["High"]["RELIANCE.NS"]
data["Low"] = df["Low"]["RELIANCE.NS"]
data["Close"] = df["Close"]["RELIANCE.NS"]
data["Volume"] = df["Volume"]["RELIANCE.NS"]

# Create features using historical information
data["Daily_Return"] = data["Close"].pct_change()
data["MA_5"] = data["Close"].rolling(window=5).mean()
data["MA_20"] = data["Close"].rolling(window=20).mean()

# Target: Will the next trading day's closing price rise?
data["Target"] = (
    data["Close"].shift(-1) > data["Close"]
).astype(int)

# Remove rows without enough historical data
# and remove the final row because its next-day target is unknown
data = data.iloc[:-1].dropna().copy()

# Save the prepared dataset
data.to_csv("data/prepared_stock_data.csv")

print("Data preparation completed!")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nTarget distribution:")
print(data["Target"].value_counts().rename(
    index={0: "DOWN_OR_SAME", 1: "UP"}
))

print("\nSaved: data/prepared_stock_data.csv")
