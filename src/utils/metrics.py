# utils/metrics.py
import pandas as pd
import numpy as np
from typing import Dict

def calc_metrics(daily_ret: pd.Series) -> Dict[str, float]:
    """
    Calculate strategy performance metrics
    
    Args:
        daily_ret: Daily return series
        
    Returns:
        Dictionary containing performance metrics
    """
    if daily_ret.empty or len(daily_ret) < 2:
        return {
            "CAGR": 0,
            "Sharpe": 0,
            "Sortino": 0,
            "MaxDD": 0,
            "WinRate": 0
        }
    
    # Convert to log returns for cumulative calculations
    log_rets = np.log1p(daily_ret)
    cum_log_rets = log_rets.cumsum()
    
    # Calculate cumulative returns without overflow
    cum_rets = np.exp(cum_log_rets) - 1
    
    # Time calculations
    total_days = (daily_ret.index[-1] - daily_ret.index[0]).days
    years = total_days / 365.0 if total_days > 0 else 1.0
    
    # CAGR using log returns
    total_return = cum_rets.iloc[-1]
    cagr = (1 + total_return) ** (1/years) - 1 if total_return > -1 else -1
    
    # Risk metrics
    annual_factor = np.sqrt(252)  # Using trading days
    ret_mean = daily_ret.mean() * 252
    ret_std = daily_ret.std() * annual_factor
    
    # Sharpe Ratio
    risk_free = 0.02  # Assumed risk-free rate
    sharpe = (ret_mean - risk_free) / ret_std if ret_std > 0 else 0
    
    # Sortino Ratio
    downside_ret = daily_ret[daily_ret < 0]
    downside_std = downside_ret.std() * annual_factor
    sortino = (ret_mean - risk_free) / downside_std if downside_std > 0 else 0
    
    # Maximum Drawdown using log returns
    rolling_max = cum_rets.expanding(min_periods=1).max()
    drawdowns = (cum_rets - rolling_max) / (1 + rolling_max)
    max_dd = drawdowns.min()
    
    # Win Rate
    win_rate = len(daily_ret[daily_ret > 0]) / len(daily_ret)
    
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "WinRate": win_rate
    }