
import os
import pickle
import pandas as pd

# Load prepared stock data
data_path = os.path.join("data", "prepared_stock_data.csv")
df = pd.read_csv(data_path)

# Load trained model
model_path = "stock_model.pkl"

if not os.path.exists(model_path):
    print("Model not found! First run: python train_model.py")
    raise SystemExit(1)

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Features must match the training script
features = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "MA_5", "MA_20"
]

# Use the latest available row
latest_data = df[features].dropna().iloc[[-1]]

prediction = int(model.predict(latest_data)[0])

print("\n===== STOCK TREND PREDICTION =====")
print("Latest available closing price:", round(float(latest_data["Close"].iloc[0]), 2))

if prediction == 1:
    print("Predicted next-day trend: UP 📈")
else:
    print("Predicted next-day trend: DOWN 📉")

print("\nNote: This is a model estimate, not a guaranteed prediction.")
