
# === Prophet Forecast for NVIDIA === #
from prophet import Prophet
import pandas as pd
import holidays
import os

# === Set Working Directory & Load CSV === #
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
data = pd.read_csv("nvidia.csv")

# === Step 1: Convert & Rename Columns for Prophet === #
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# ✅ Use 'Close' as target for forecasting
if 'Close' not in data.columns:
    raise ValueError("Missing 'Close' column in CSV.")

data = data.rename(columns={"Date": "ds", "Close": "y"})
data = data[['ds', 'y']].dropna()
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data.dropna(inplace=True)

# === Step 2: Add Indian Holidays === #
years = pd.DatetimeIndex(data["ds"]).year.unique()
ind_holidays = holidays.India(years=years)
holiday_df = pd.DataFrame({
    "ds": pd.to_datetime(list(ind_holidays.keys())),
    "holiday": "india_national"
})

# === Step 3: Train Prophet Model === #
model = Prophet(holidays=holiday_df)
model.fit(data)

# === Step 4: Forecast 30 Days Ahead === #
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# === Step 5: Save Last 30 Days of Forecast === #
forecast[['ds', 'yhat']].tail(30).to_csv("forecast_prophet.csv", index=False)

print("\n✅ Forecast saved as 'forecast_prophet.csv'")
