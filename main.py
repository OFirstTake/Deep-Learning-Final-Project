from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os

app = FastAPI(title="Smart Stock Prediction API (Vietnam Tech)")

# The Vietnamese companies list in technology
VN_technology = [
    "CKV", "CMG", "CMT", "ICT", "ELC", "FPT", "HIG", "HPT", "ITD", "KST", 
    "LTC", "ONE", "PMJ", "PMT", "POT", "SAM", "SBD", "SGT", "SMT", "SRA", 
    "SRB", "ST8", "TST", "UNI", "VAT", "VEC", "VIE", "VLA", "VTC", "VTE"
]

# Load the trained model from Task 2
model = tf.keras.models.load_model('model_2_1.keras')

# Define input structure: Require the user to input Ticker and Date
class StockQuery(BaseModel):
    ticker: str
    date: str 

@app.post("/predict_stock")
def predict_stock(data: StockQuery):
    try:
        # Check if the requested ticker is in the supported list
        ticker = data.ticker.upper()
        if ticker not in VN_technology:
            raise HTTPException(status_code=400, detail=f"Stock ticker {ticker} is not in the supported list!")

        # Automatically find the CSV file for the ticker in the specified directory
        matching_files = glob.glob(f"data-vn-20230228\\stock-historical-data\\{ticker}-*.csv")
        
        if not matching_files:
            raise HTTPException(status_code=404, detail=f"CSV file for ticker {ticker} not found. Please check the data directory!")
            
        file_path = matching_files[0]
        
        # Read and clean the data following the Notebook's preprocessing steps
        df = pd.read_csv(file_path, on_bad_lines='skip')
        if 'Trading Date' not in df.columns and 'TradingDate' in df.columns:
            df.rename(columns={'TradingDate': 'Trading Date'}, inplace=True)
            
        if df.isna().values.any():
            df = df.interpolate(method='linear', limit_direction='both', numeric_only=True)

        df['Trading Date'] = pd.to_datetime(df['Trading Date'])
        df = df.sort_values('Trading Date').reset_index(drop=True)

        # Find the index of the target date provided by the user
        target_date = pd.to_datetime(data.date)
        if target_date not in df['Trading Date'].values:
            raise HTTPException(status_code=404, detail=f"No trading data available for date {data.date} of ticker {ticker}!")

        target_idx = df[df['Trading Date'] == target_date].index[0]

        if target_idx < 30:
            raise HTTPException(status_code=400, detail="Not enough 30 days of historical data to make a prediction!")

        # Extract the sequence of the previous 30 days
        window_df = df.iloc[target_idx - 30 : target_idx]
        features = window_df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(float)
        
        # Data Normalization (MinMax Scale)
        X_input = features.reshape(1, 30, 5)
        min_f = np.min(X_input[0], axis=0)
        max_f = np.max(X_input[0], axis=0)
        range_f = max_f - min_f
        range_f[range_f == 0] = 1.0 # Prevent division by zero
        
        X_input_norm = (X_input - min_f) / range_f

        # Predict the Open price for the target date using the loaded model
        y_pred_norm = model.predict(X_input_norm)

        # Denormalize the predicted Open price
        predicted_open_price = (y_pred_norm[0][0] * range_f[0]) + min_f[0]
        actual_open_price = float(df.iloc[target_idx]['Open'])

        return {
            "ticker": ticker,
            "target_date": data.date,
            "predicted_open_price_VND": round(float(predicted_open_price), 2),
            "actual_open_price_VND": actual_open_price,
            "difference_VND": round(float(predicted_open_price - actual_open_price), 2)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))