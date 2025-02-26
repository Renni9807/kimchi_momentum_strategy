# src/analysis/strategy_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from utils.config import TradingConfig

class StrategyAnalyzer:
    """Kimchi Momentum Strategy Analysis Class"""
    
    def __init__(self, price_df: pd.DataFrame, funding_df: pd.DataFrame, config: TradingConfig):
        """
        Initialize analyzer with data and configuration
        
        Args:
            price_df: DataFrame with upbit and binance prices
            funding_df: DataFrame with funding rates
            config: Trading configuration
        """
        self.price_df = price_df
        self.funding_df = funding_df
        self.config = config
        
    def analyze_signal_generation(self, X: float, Y: float) -> pd.DataFrame:
        """
        Analyze signal generation logic
        
        Args:
            X: Up threshold percentage
            Y: Down threshold percentage
        """
        data = self.price_df.copy()
        data['upbit_pct_change'] = data['upbit_close'].pct_change() * 100
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            if signals[i-1] == 0:  # No position
                if data['upbit_pct_change'].iloc[i] >= X * 100:
                    signals[i] = 1  # Long
                elif data['upbit_pct_change'].iloc[i] <= -Y * 100:
                    signals[i] = -1  # Short
            else:
                signals[i] = signals[i-1]  # Hold position
        
        data['signals'] = signals
        return self._analyze_signals(data)
    
    def _analyze_signals(self, data: pd.DataFrame) -> Dict:
        """Analyze signal statistics"""
        signal_stats = {
            'long_signals': len(data[data['signals'] == 1]),
            'short_signals': len(data[data['signals'] == -1]),
            'no_position': len(data[data['signals'] == 0]),
            'avg_position_duration': self._calculate_avg_position_duration(data['signals'])
        }
        return signal_stats
    
    def analyze_returns(self, signals: np.ndarray) -> Dict:
        """Analyze strategy returns"""
        data = self.price_df.copy()
        data['signals'] = signals
        data['binance_returns'] = data['binance_close'].pct_change()
        data['strategy_returns'] = data['signals'] * self.config.leverage * data['binance_returns']
        
        return {
            'mean_daily_return': data['strategy_returns'].mean() * 100,
            'return_std': data['strategy_returns'].std() * 100,
            'max_gain': data['strategy_returns'].max() * 100,
            'max_loss': data['strategy_returns'].min() * 100,
            'sharpe_components': self._calculate_sharpe_components(data['strategy_returns'])
        }
    
    def _calculate_sharpe_components(self, returns: pd.Series) -> Dict:
        """Calculate components of Sharpe ratio"""
        annual_factor = np.sqrt(252)
        returns_mean = returns.mean() * 252
        returns_std = returns.std() * annual_factor
        sharpe = returns_mean / returns_std if returns_std > 0 else 0
        
        return {
            'annualized_return': returns_mean * 100,
            'annualized_volatility': returns_std * 100,
            'sharpe_ratio': sharpe
        }
    
    def _calculate_avg_position_duration(self, signals: pd.Series) -> float:
        """Calculate average duration of positions"""
        position_changes = signals.diff().fillna(0)
        entry_points = position_changes != 0
        durations = []
        current_duration = 0
        
        for change in position_changes:
            if change != 0:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 1
            else:
                current_duration += 1
                
        return np.mean(durations) if durations else 0
    
    def verify_data_split(self, split_ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Verify train/test data split"""
        split_idx = int(len(self.price_df) * split_ratio)
        train = self.price_df.iloc[:split_idx]
        test = self.price_df.iloc[split_idx:]
        
        split_info = {
            'train_period': f"{train.index[0]} ~ {train.index[-1]}",
            'test_period': f"{test.index[0]} ~ {test.index[-1]}",
            'train_samples': len(train),
            'test_samples': len(test)
        }
        
        print("\nData Split Analysis:")
        print("-" * 50)
        for key, value in split_info.items():
            print(f"{key}: {value}")
        
        return train, test

# Example usage in kimchi_main.py:
"""
from analysis.strategy_analyzer import StrategyAnalyzer

def main():
    # Load data and config as before...
    
    # Create analyzer instance
    analyzer = StrategyAnalyzer(price_df, funding_df, config)
    
    # Analyze current best parameters
    signal_stats = analyzer.analyze_signal_generation(X=0.01, Y=0.025)
    print("\nSignal Generation Analysis:")
    print(signal_stats)
    
    # Verify data split
    train_df, test_df = analyzer.verify_data_split()
"""