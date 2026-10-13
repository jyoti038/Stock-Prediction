
import yfinance as yf
import os

stock_symbol = "RELIANCE.NS"

print("Downloading stock market data...")

data = yf.download(
    stock_symbol,
    period="5y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if data.empty:
    print("No data downloaded. Check your internet connection.")
else:
    os.makedirs("data", exist_ok=True)

    file_path = "data/stock_data.csv"
    data.to_csv(file_path)

    print("Data downloaded successfully!")
    print("Total rows:", len(data))
    print("Saved file:", file_path)
    print(data.tail())
