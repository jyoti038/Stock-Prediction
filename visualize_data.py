
import pandas as pd
import matplotlib.pyplot as plt

# Read stock data
df = pd.read_csv("data/stock_data.csv", header=[0, 1], index_col=0)

# Extract the closing price
close_price = df["Close"]["RELIANCE.NS"]

# Convert index into dates
df.index = pd.to_datetime(df.index)

# Plot closing prices
plt.figure(figsize=(12, 6))
plt.plot(df.index, close_price, label="Reliance Closing Price")

plt.title("Reliance Industries - Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Adjusted Stock Price (₹)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
