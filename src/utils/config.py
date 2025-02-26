# utils/config.py
from dataclasses import dataclass

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    # basic setting
    initial_capital: float = 100000  # Initial capital in USD
    leverage: float = 3.0           # Leverage used
    position_size: float = 0.95     # Percentage of capital to use per trade
    
    # Fee structure
    maker_fee: float = 0.0002      # 0.02% maker fee
    taker_fee: float = 0.0004      # 0.04% taker fee
    slippage: float = 0.0005       # 0.05% assumed slippage
    
    # Risk management
    max_position_usd: float = 1000000  # Maximum position size in USD
    daily_stop_loss: float = 0.03      # 3% daily stop loss
    
    # Technical Indicators
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9