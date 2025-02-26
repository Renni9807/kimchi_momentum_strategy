import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_funding_data():
    # Start date matching with your price data
    # You might need to adjust this based on your actual price data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Creating timestamps (8-hour intervals as funding rates are typically every 8 hours)
    timestamps = []
    current_date = start_date
    while current_date <= end_date:
        timestamps.append(current_date)
        current_date += timedelta(hours=8)
    
    # Generate sample funding rates (typically between -0.1% to 0.1%)
    funding_rates = np.random.normal(0, 0.0005, len(timestamps))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'funding_rate': funding_rates
    })
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Save to CSV
    df.to_csv('data/raw/binance_funding.csv')
    print("Sample funding rate data has been created at data/raw/binance_funding.csv")
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Number of records: {len(df)}")
    
if __name__ == "__main__":
    create_sample_funding_data()