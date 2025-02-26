# utils/technical_indicators.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional

class TechnicalIndicators:
    def __init__(self, data: pd.Series):
        """
        Initialize technical indicators calculator
        
        Args:
            data: Price series data
        """
        self.data = data
        self._rsi = None
        self._macd = None
        self._bb = None
        self._last_calc_index = None
    
    def _needs_update(self) -> bool:
        """Check if indicators need to be recalculated"""
        return (self._last_calc_index is None or 
                len(self.data) != self._last_calc_index)
    
    @property
    def rsi(self, period: int = 14) -> pd.Series:
        """Calculate RSI if needed and return cached result"""
        if self._needs_update() or self._rsi is None:
            delta = self.data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self._rsi = 100 - (100 / (1 + rs))
            self._last_calc_index = len(self.data)
        return self._rsi
    
    @property
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD if needed and return cached result"""
        if self._needs_update() or self._macd is None:
            exp1 = self.data.ewm(span=fast, adjust=False).mean()
            exp2 = self.data.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            self._macd = (macd_line, signal_line, histogram)
            self._last_calc_index = len(self.data)
        return self._macd
    
    @property
    def bollinger_bands(self, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands if needed and return cached result"""
        if self._needs_update() or self._bb is None:
            middle = self.data.rolling(window=period).mean()
            std_dev = self.data.rolling(window=period).std()
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            self._bb = (middle, upper, lower)
            self._last_calc_index = len(self.data)
        return self._bb
    
    def update(self, new_data: float) -> None:
        """
        Update the price series with new data
        
        Args:
            new_data: New price point to add
        """
        self.data = pd.concat([self.data, pd.Series([new_data])])
        # Force recalculation on next access
        self._last_calc_index = None
    
    def reset_cache(self) -> None:
        """Reset all cached calculations"""
        self._rsi = None
        self._macd = None
        self._bb = None
        self._last_calc_index = None
def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    ATR(Average True Range) 계산
    """
    if 'binance_high' not in data.columns or 'binance_low' not in data.columns:
        data['binance_high'] = data['binance_close']
        data['binance_low'] = data['binance_close']
    
    high = data['binance_high']
    low = data['binance_low']
    close = data['binance_close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr

def calculate_adx(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    ADX(Average Directional Index) 계산 - 데이터 타입 문제 해결
    """
    if 'binance_high' not in data.columns or 'binance_low' not in data.columns:
        data['binance_high'] = data['binance_close']
        data['binance_low'] = data['binance_close']
    
    high = data['binance_high'].astype(float)
    low = data['binance_low'].astype(float)
    close = data['binance_close'].astype(float)
    
    high_diff = high - high.shift(1)
    low_diff = low.shift(1) - low
    
    pos_dm = pd.Series(0.0, index=high.index)  # float 
    neg_dm = pd.Series(0.0, index=high.index)  # float 
    
    pos_dm_mask = (high_diff > 0) & (high_diff > low_diff)
    neg_dm_mask = (low_diff > 0) & (low_diff > high_diff)
    
    pos_dm[pos_dm_mask] = high_diff[pos_dm_mask]
    neg_dm[neg_dm_mask] = low_diff[neg_dm_mask]
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_smooth = tr.rolling(window=window, min_periods=1).mean()
    pos_dm_smooth = pos_dm.rolling(window=window, min_periods=1).mean()
    neg_dm_smooth = neg_dm.rolling(window=window, min_periods=1).mean()
    
    pos_di = 100 * pos_dm_smooth / tr_smooth
    neg_di = 100 * neg_dm_smooth / tr_smooth
    
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di).replace(0, np.nan)
    adx = dx.rolling(window=window, min_periods=1).mean()
    
    return adx