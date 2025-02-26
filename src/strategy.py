import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KimchiStrategy:
    """
    Kimchi Premium 전략 클래스
    - Upbit 가격이 Binance 대비 X% 이상 상승하면 Long
    - Upbit 가격이 Binance 대비 Y% 이상 하락하면 Short
    """
    def __init__(self, 
                 long_threshold: float, 
                 short_threshold: float,
                 binance_maker_fee: float = 0.0002,  # 0.02%
                 binance_taker_fee: float = 0.0004,  # 0.04%
                 upbit_fee: float = 0.0005,          # 0.05%
                 funding_fee_threshold: float = 0.01  # 1% 이상이면 포지션 진입 제한
                 ):
        """
        Args:
            long_threshold (float): Long 진입을 위한 Kimchi Premium 임계값 (X%)
            short_threshold (float): Short 진입을 위한 Kimchi Premium 임계값 (Y%)
            binance_maker_fee (float): Binance Maker 수수료
            binance_taker_fee (float): Binance Taker 수수료
            upbit_fee (float): Upbit 거래 수수료
            funding_fee_threshold (float): Funding Fee 임계값
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.binance_maker_fee = binance_maker_fee
        self.binance_taker_fee = binance_taker_fee
        self.upbit_fee = upbit_fee
        self.funding_fee_threshold = funding_fee_threshold
        
    def calculate_kimchi_premium(self, upbit_price: float, binance_price: float) -> float:
        """Kimchi Premium 계산"""
        return ((upbit_price / binance_price) - 1) * 100
        
    def calculate_effective_premium(self, kimchi_premium: float, position_type: str) -> float:
        """
        거래 비용을 고려한 실효 프리미엄 계산
        
        Args:
            kimchi_premium (float): 원래의 Kimchi Premium
            position_type (str): 'long' 또는 'short'
            
        Returns:
            float: 실효 프리미엄
        """
        # 거래 비용 계산 (양방향 거래이므로 2배)
        total_fee = (self.binance_taker_fee + self.upbit_fee) * 2
        
        # Long의 경우 비용을 차감, Short의 경우 비용을 가산
        if position_type == 'long':
            return kimchi_premium - total_fee * 100  # percentage로 변환
        else:
            return kimchi_premium + total_fee * 100
            
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 신호 생성"""
        try:
            # DataFrame 복사본 생성
            result_df = df.copy()
            
            # Kimchi Premium 계산
            result_df.loc[:, 'kimchi_premium'] = result_df.apply(
                lambda x: self.calculate_kimchi_premium(x['upbit_close'], x['binance_close']), 
                axis=1
            )
            
            # 실효 프리미엄 계산
            result_df.loc[:, 'effective_premium_long'] = result_df['kimchi_premium'].apply(
                lambda x: self.calculate_effective_premium(x, 'long')
            )
            result_df.loc[:, 'effective_premium_short'] = result_df['kimchi_premium'].apply(
                lambda x: self.calculate_effective_premium(x, 'short')
            )
            
            # 기본적으로 모든 포지션을 0(중립)으로 초기화
            result_df.loc[:, 'position'] = 0
            
            # Long/Short 조건에 따라 포지션 설정
            result_df.loc[result_df['effective_premium_long'] > self.long_threshold, 'position'] = 1
            result_df.loc[result_df['effective_premium_short'] < -self.short_threshold, 'position'] = -1
            
            return result_df
                
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
            
    def calculate_pnl(self, row: pd.Series) -> float:
        """
        각 거래의 수익률 계산 (거래 비용 포함)
        """
        pnl = row['strategy_returns'] if not pd.isna(row['strategy_returns']) else 0
        
        # 포지션 변경이 있을 경우 거래 비용 차감
        if not pd.isna(row['position_change']) and row['position_change'] != 0:
            pnl -= (self.binance_taker_fee + self.upbit_fee)
            
        # Funding Fee 차감 (8시간마다)
        if 'funding_rate' in row.index and not pd.isna(row['funding_rate']):
            pnl -= row['funding_rate']
            
        return pnl
            
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Tuple[pd.DataFrame, Dict]:
        """전략 백테스트 실행"""
        try:
            # DataFrame 복사본 생성
            result_df = df.copy()
            
            # 수익률 계산
            result_df.loc[:, 'binance_returns'] = result_df['binance_close'].pct_change()
            result_df.loc[:, 'position_change'] = result_df['position'].diff()
            result_df.loc[:, 'strategy_returns'] = result_df['position'].shift(1) * result_df['binance_returns']
            result_df.loc[:, 'final_returns'] = result_df.apply(self.calculate_pnl, axis=1)
            
            # 누적 수익률 계산
            result_df.loc[:, 'cumulative_returns'] = (1 + result_df['final_returns']).cumprod()
            result_df.loc[:, 'cumulative_value'] = initial_capital * result_df['cumulative_returns']
            
            # 성과 지표 계산 (이하 동일)
            total_days = (result_df.index[-1] - result_df.index[0]).days
            total_years = total_days / 365.0
            
            final_value = float(result_df['cumulative_value'].iloc[-1])  # numpy.int64를 float으로 변환
            cagr = float((final_value / initial_capital) ** (1 / total_years) - 1)
            
            historical_max = result_df['cumulative_value'].expanding().max()
            drawdowns = result_df['cumulative_value'] / historical_max - 1
            max_drawdown = float(drawdowns.min())
            
            annual_volatility = float(result_df['final_returns'].std() * np.sqrt(365))
            sharpe_ratio = float((cagr - 0) / annual_volatility if annual_volatility != 0 else 0)
            
            total_trades = int(result_df['position_change'].ne(0).sum())  # int로 변환
            win_rate = float(result_df['final_returns'].gt(0).mean())
            
            metrics = {
                'CAGR': cagr,
                'Max_Drawdown': max_drawdown,
                'Sharpe_Ratio': sharpe_ratio,
                'Total_Return': float((final_value / initial_capital) - 1),
                'Annual_Volatility': annual_volatility,
                'Total_Trades': total_trades,
                'Win_Rate': win_rate
            }
            
            return result_df, metrics
                
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise
            
    def optimize_parameters(self, 
                          df: pd.DataFrame, 
                          long_thresholds: List[float], 
                          short_thresholds: List[float]) -> Tuple[float, float, float, pd.DataFrame]:
        """최적의 파라미터 탐색"""
        best_sharpe = -np.inf
        best_params = (0, 0)
        results = []
        
        for long_thresh in long_thresholds:
            for short_thresh in short_thresholds:
                strategy = KimchiStrategy(
                    long_thresh, 
                    short_thresh,
                    self.binance_maker_fee,
                    self.binance_taker_fee,
                    self.upbit_fee,
                    self.funding_fee_threshold
                )
                
                df_signals = strategy.generate_signals(df.copy())
                _, metrics = strategy.backtest(df_signals)
                
                sharpe = metrics['Sharpe_Ratio']
                results.append({
                    'Long_Threshold': long_thresh,
                    'Short_Threshold': short_thresh,
                    'Sharpe_Ratio': sharpe
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (long_thresh, short_thresh)
                    
        results_df = pd.DataFrame(results)
        results_df['Sharpe_Ratio'] = results_df['Sharpe_Ratio'].round(2)
        
        return best_params[0], best_params[1], best_sharpe, results_df