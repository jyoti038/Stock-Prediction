
import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("===== STOCK MARKET TREND PREDICTION =====")

# Load prepared stock data
file_path = os.path.join("data", "prepared_stock_data.csv")
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Total records:", len(df))

# Features used for prediction
features = ["Open", "High", "Low", "Close", "Volume",
            "Daily_Return", "MA_5", "MA_20"]

target = "Target"

# Keep only required columns and remove missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target].astype(int)

# Split data chronologically: earlier records for training,
# later records for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL RESULTS =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save trained model
model_path = "stock_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully:", os.path.abspath(model_path))
print("SUCCESS: Model training completed!")
