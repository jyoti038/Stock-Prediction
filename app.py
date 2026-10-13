
import os
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Stock Market Trend Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Trend Prediction")
st.caption("Big Data Analytics + Machine Learning")
st.divider()

DATA_PATH = os.path.join("data", "prepared_stock_data.csv")
MODEL_PATH = "stock_model.pkl"

if not os.path.exists(DATA_PATH):
    st.error("Prepared stock data not found. Check data/prepared_stock_data.csv")
    st.stop()

df = pd.read_csv(DATA_PATH)

required_features = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "MA_5", "MA_20"
]

if df.empty or "Close" not in df.columns:
    st.error("The dataset is empty or does not contain a Close column.")
    st.stop()

# Summary metrics
latest_close = float(df["Close"].iloc[-1])
average_close = float(df["Close"].mean())
highest_close = float(df["Close"].max())
lowest_close = float(df["Close"].min())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Closing Price", f"₹{latest_close:,.2f}")
c2.metric("Average Close", f"₹{average_close:,.2f}")
c3.metric("Highest Close", f"₹{highest_close:,.2f}")
c4.metric("Lowest Close", f"₹{lowest_close:,.2f}")

st.subheader("Historical Closing Price")

chart_data = df[["Close"]].copy()
chart_data["Record"] = range(1, len(chart_data) + 1)
st.line_chart(chart_data.set_index("Record")["Close"])

st.subheader("Stock Dataset")
st.write(f"Total records: **{len(df):,}**")
st.dataframe(df.tail(10), use_container_width=True)

st.subheader("🤖 Machine Learning Prediction")

if os.path.exists(MODEL_PATH):
    if all(col in df.columns for col in required_features):
        try:
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)

            latest = df[required_features].dropna().iloc[[-1]]
            prediction = int(model.predict(latest)[0])

            if prediction == 1:
                st.success("Predicted trend: UP 📈")
            else:
                st.warning("Predicted trend: DOWN 📉")

            st.caption(
                "This is a model estimate based on historical data, "
                "not a guaranteed prediction or investment advice."
            )
        except Exception as e:
            st.error(f"Could not generate prediction: {e}")
    else:
        st.error("Some required model features are missing from the dataset.")
else:
    st.warning("Trained model not found. Run train_model.py first.")

st.divider()
st.caption("Academic project | Stock Market Trend Prediction with Big Data")
