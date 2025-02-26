# utils/data.py
import pandas as pd
from typing import Tuple
from pathlib import Path

def load_data(data_path: str = "../data/raw/") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price and funding rate data
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple containing (price_df, funding_df)
    """
    # Load price data
    upbit_df = pd.read_csv(f"{data_path}upbit_price.csv", 
                          index_col="timestamp", 
                          parse_dates=True)
    binance_df = pd.read_csv(f"{data_path}binance_perpetual.csv", 
                            index_col="timestamp", 
                            parse_dates=True)
    
    # Combine price data
    price_df = pd.DataFrame({
        "upbit_close": upbit_df["close"],
        "binance_close": binance_df["close"]
    }).dropna()
    
    # Load funding rate data
    funding_df = pd.read_csv(f"{data_path}binance_funding.csv", 
                           index_col="timestamp", 
                           parse_dates=True)
    
    # Ensure all timestamps are tz-aware (UTC)
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize('UTC')
    if funding_df.index.tz is None:
        funding_df.index = funding_df.index.tz_localize('UTC')
    
    return price_df.sort_index(), funding_df.sort_index()

def split_data(df: pd.DataFrame, ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and test sets
    
    Args:
        df: Input dataframe
        ratio: Train set ratio (default: 0.5)
        
    Returns:
        Tuple containing (train_df, test_df)
    """
    split_idx = int(len(df) * ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]