# kimchi_main.py

import numpy as np
import pandas as pd
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, List
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.data import load_data, split_data
from utils.metrics import calc_metrics
from utils.technical_indicators import calculate_atr, calculate_adx
from utils.config import TradingConfig
from utils.funding_generator import create_funding_scenarios
from visualization.plots import plot_sharpe_heatmap, plot_equity_curve

def calculate_directional_volatility(data: pd.DataFrame, window: int = 30) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate separate volatility for uptrends and downtrends.
    """
    returns = data['binance_close'].pct_change()
    
    up_returns = returns.copy()
    up_returns[returns <= 0] = np.nan
    
    down_returns = returns.copy()
    down_returns[returns >= 0] = np.nan
    
    up_vol = up_returns.rolling(window, min_periods=1).std() * np.sqrt(252)
    down_vol = down_returns.rolling(window, min_periods=1).std() * np.sqrt(252)
    
    return up_vol, down_vol

def calculate_trend_strength(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate trend strength using ATR and ADX.
    """
    atr = calculate_atr(data, window)
    adx = calculate_adx(data, window)
    # Here we normalize ADX by (ATR * 100) as an example
    return adx / (atr * 100)

def detect_extreme_market(data: pd.DataFrame, row_idx: int) -> bool:
    """
    Detect extreme market conditions.
    """
    if row_idx < 30:  # Not enough data
        return False
        
    current_data = data.iloc[row_idx]
    hist_data = data.iloc[max(0, row_idx - 30):row_idx]
    
    conditions = [
        # 1. High volatility (annualized volatility > 100%)
        current_data['vol_30d'] > 1.0,
        # 2. Rapid price change (±10% in one bar)
        abs(current_data['binance_close'] / data['binance_close'].iloc[row_idx - 1] - 1) > 0.1,
        # 3. Large discrepancy between exchanges (±5%)
        abs(current_data['upbit_close'] / current_data['binance_close'] - 1) > 0.05,
        # 4. Big change vs. recent high/low over the past 30 bars (±20% above or below)
        current_data['binance_close'] < hist_data['binance_close'].min() * 0.8,
        current_data['binance_close'] > hist_data['binance_close'].max() * 1.2
    ]
    
    return any(conditions)

def dynamic_stop_loss(position_type: str, vol_30d: float, trend_strength: float,
                      up_vol: float, down_vol: float, config: TradingConfig) -> float:
    """
    Dynamic stop-loss calculation.
    """
    base_stop = config.daily_stop_loss
    
    if position_type == 'short':
        base_stop *= 0.7  # Tighter stop for short
        vol_factor = up_vol / 0.4
    else:  # long
        vol_factor = down_vol / 0.4
    
    vol_adjustment = 0.5 / max(vol_factor, 0.5)
    trend_adjustment = 1 + (trend_strength * 0.5)
    
    final_stop = base_stop * vol_adjustment * trend_adjustment
    return max(final_stop, config.daily_stop_loss * 0.5)

def compute_position_size(capital: float, price: float, position_type: str,
                          vol_30d: float, funding_rate: float, up_vol: float, 
                          down_vol: float, trend_strength: float, 
                          is_extreme_market: bool, config: TradingConfig) -> float:
    """
    Compute position size based on risk factors.
    """
    max_pos = min(capital * config.leverage * config.position_size, 
                  config.max_position_usd)
    
    # Volatility-based adjustment
    if position_type == 'long':
        vol_scale = 0.8 / (down_vol / 0.5)
    else:
        vol_scale = 0.8 / (up_vol / 0.5)
    max_pos *= np.clip(vol_scale, 0.2, 0.8)
    
    # Funding rate consideration
    if position_type == 'long':
        max_pos *= 0.8
        if funding_rate > 0:
            max_pos *= np.exp(-7 * funding_rate)
    else:  # short
        max_pos *= 0.7
        if funding_rate < 0:
            max_pos *= np.exp(7 * funding_rate)
    
    # Trend strength consideration
    trend_scale = 1 + (trend_strength * 0.3)
    max_pos *= np.clip(trend_scale, 0.5, 1.2)
    
    # Extreme market condition
    if is_extreme_market:
        max_pos *= 0.3
    
    return min(max_pos / price, config.max_position_usd / price)

def compute_daily_strategy_returns(df: pd.DataFrame, funding_df: pd.DataFrame, 
                                   X: float, Y: float, config: TradingConfig) -> pd.Series:
    """
    Compute daily strategy returns (here, X=0.01 means ~1% price change trigger).
    """
    data = df.copy()
    
    # 1) Upbit change in % (remove *100 to interpret 0.01 as 1%)
    data['upbit_pct_change'] = data['upbit_close'].pct_change()
    
    data['vol_30d'] = data['binance_close'].pct_change().rolling(30).std() * np.sqrt(252)
    data['up_vol'], data['down_vol'] = calculate_directional_volatility(data)
    data['trend_strength'] = calculate_trend_strength(data)
    
    signals = np.zeros(len(data))
    position_sizes = np.zeros(len(data))
    capital = config.initial_capital
    
    price_returns = []
    funding_returns = []
    
    for i in range(1, len(data)):
        is_extreme = detect_extreme_market(data, i)
        current_time = data.index[i]
        
        # Current funding rate
        current_funding = 0
        if current_time in funding_df.index:
            current_funding = funding_df.loc[current_time, 'funding_rate']
        
        # Generate signals (only when previous signal is 0)
        if signals[i - 1] == 0:
            # upbit_pct_change >= X -> long
            if data['upbit_pct_change'].iloc[i] >= X:
                signals[i] = 1
                position_sizes[i] = compute_position_size(
                    capital=capital,
                    price=data['binance_close'].iloc[i],
                    position_type='long',
                    vol_30d=data['vol_30d'].iloc[i],
                    funding_rate=current_funding,
                    up_vol=data['up_vol'].iloc[i],
                    down_vol=data['down_vol'].iloc[i],
                    trend_strength=data['trend_strength'].iloc[i],
                    is_extreme_market=is_extreme,
                    config=config
                )
            # upbit_pct_change <= -Y -> short
            elif data['upbit_pct_change'].iloc[i] <= -Y:
                signals[i] = -1
                position_sizes[i] = compute_position_size(
                    capital=capital,
                    price=data['binance_close'].iloc[i],
                    position_type='short',
                    vol_30d=data['vol_30d'].iloc[i],
                    funding_rate=current_funding,
                    up_vol=data['up_vol'].iloc[i],
                    down_vol=data['down_vol'].iloc[i],
                    trend_strength=data['trend_strength'].iloc[i],
                    is_extreme_market=is_extreme,
                    config=config
                )
        else:
            # Maintain previous position
            signals[i] = signals[i - 1]
            position_sizes[i] = position_sizes[i - 1]
            
            # Dynamic stop-loss check
            daily_return = data['binance_close'].pct_change().iloc[i]
            stop_loss = dynamic_stop_loss(
                position_type='long' if signals[i] > 0 else 'short',
                vol_30d=data['vol_30d'].iloc[i],
                trend_strength=data['trend_strength'].iloc[i],
                up_vol=data['up_vol'].iloc[i],
                down_vol=data['down_vol'].iloc[i],
                config=config
            )
            if daily_return * signals[i] < -stop_loss:
                signals[i] = 0
                position_sizes[i] = 0
        
        # Calculate price return
        price_return = signals[i - 1] * position_sizes[i - 1] * data['binance_close'].pct_change().iloc[i]
        price_returns.append(price_return)
        
        # Funding return
        if current_time in funding_df.index:
            funding_return = -signals[i - 1] * position_sizes[i - 1] * current_funding
            funding_returns.append(funding_return)
        else:
            funding_returns.append(0.0)
        
        # Update capital
        capital *= (1 + price_returns[-1] + funding_returns[-1])
    
    # Convert to daily returns (if data is hourly, sum up per day)
    returns_df = pd.DataFrame({
        'price_returns': price_returns,
        'funding_returns': funding_returns
    }, index=data.index[1:])
    
    strategy_daily = (returns_df['price_returns'] + returns_df['funding_returns']).resample('D').sum().fillna(0)
    return strategy_daily

def parallel_monte_carlo_optimization(args: Tuple) -> Tuple:
    """
    Helper function for parallel Monte Carlo optimization.
    """
    X, Y, price_df, funding_df, config = args
    daily_rets = compute_daily_strategy_returns(price_df, funding_df, X, Y, config)
    metrics = calc_metrics(daily_rets)
    return (X, Y, metrics['Sharpe'])

def process_monte_carlo_results(results: defaultdict, X_list: np.ndarray, 
                                Y_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Process Monte Carlo simulation results.
    """
    results_matrix = np.zeros((len(X_list), len(Y_list)))
    results_std = np.zeros((len(X_list), len(Y_list)))
    best_sharpe = -np.inf
    best_params = None
    
    for i, X in enumerate(X_list):
        for j, Y in enumerate(Y_list):
            sharpe_values = results[(X, Y)]
            avg_sharpe = np.mean(sharpe_values)
            std_sharpe = np.std(sharpe_values)
            
            results_matrix[i, j] = avg_sharpe
            results_std[i, j] = std_sharpe
            
            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_params = (X, Y)
    
    return results_matrix, results_std, best_params

def main():
    # Trading settings
    config = TradingConfig(
        initial_capital=100000,
        leverage=3.0,
        position_size=0.5,
        daily_stop_loss=0.02,
        max_position_usd=1000000
    )
    
    # Load data
    price_df, _ = load_data("data/raw/")
    print("Entire data range:", price_df.index.min(), "~", price_df.index.max())
    print("Total sample size:", len(price_df))
    
    # Split 50:50
    price_df_train, price_df_test = split_data(price_df, ratio=0.5)
    print("\n[Data Split]")
    print("Train range:", price_df_train.index.min(), "~", price_df_train.index.max(), f"({len(price_df_train)} rows)")
    print("Test range :", price_df_test.index.min(), "~", price_df_test.index.max(), f"({len(price_df_test)} rows)")
    
    # Funding fee scenarios
    funding_scenarios_train = create_funding_scenarios(price_df_train, n_scenarios=5)
    funding_scenarios_test = create_funding_scenarios(price_df_test, n_scenarios=5)
    
    # X, Y grid (e.g., from 1% to 5%)
    X_list = np.arange(0.01, 0.06, 0.01)
    Y_list = np.arange(0.01, 0.06, 0.01)
    
    n_cores = min(4, os.cpu_count())
    print(f"\nUsing {n_cores} CPU cores")
    
    # Parameter scenarios
    param_scenarios = [
        (x, y, price_df_train, scenario, config)
        for x in X_list
        for y in Y_list
        for scenario in funding_scenarios_train
    ]
    print("Number of scenarios:", len(param_scenarios))
    
    # Parallel Monte Carlo
    results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(parallel_monte_carlo_optimization, p) for p in param_scenarios]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                X, Y, sharpe = future.result()
                results[(X, Y)].append(sharpe)
            except Exception as e:
                print("Error occurred:", e)
    
    # Process results
    results_matrix, results_std, best_params = process_monte_carlo_results(results, X_list, Y_list)
    
    print("\n[Backtest Results]")
    print("Best Parameters: X=%.2f%%, Y=%.2f%%" % (best_params[0]*100, best_params[1]*100))
    best_i = np.where(X_list == best_params[0])[0][0]
    best_j = np.where(Y_list == best_params[1])[0][0]
    print("Average Sharpe Ratio: %.4f" % results_matrix[best_i, best_j])
    print("Sharpe Ratio Std Dev: %.4f" % results_std[best_i, best_j])
    
    # Heatmap
    save_path = Path("data/results/kimchi_strategy_heatmap.png")
    plot_sharpe_heatmap(results_matrix, X_list, Y_list, save_path, title="Train Set Sharpe Ratio")
    
    # Forward test
    print("\n[Forward Test]")
    best_X, best_Y = best_params
    forward_metrics = []
    equity_curves = []
    
    for funding_df in tqdm(funding_scenarios_test, desc="Forward tests"):
        daily_rets = compute_daily_strategy_returns(price_df_test, funding_df, best_X, best_Y, config)
        metrics = calc_metrics(daily_rets)
        forward_metrics.append(metrics)
        equity_curves.append((1 + daily_rets).cumprod())
    
    avg_sharpe = np.mean([m['Sharpe'] for m in forward_metrics])
    std_sharpe = np.std([m['Sharpe'] for m in forward_metrics])
    avg_cagr = np.mean([m['CAGR'] for m in forward_metrics])
    std_cagr = np.std([m['CAGR'] for m in forward_metrics])
    avg_maxdd = np.mean([m['MaxDD'] for m in forward_metrics])
    std_maxdd = np.std([m['MaxDD'] for m in forward_metrics])
    avg_wr = np.mean([m['WinRate'] for m in forward_metrics])
    std_wr = np.std([m['WinRate'] for m in forward_metrics])
    
    print("\n[Forward test (avg ± std)]")
    print(f"Sharpe Ratio = {avg_sharpe:.4f} (±{std_sharpe:.4f})")
    print(f"CAGR         = {avg_cagr*100:.2f}% (±{std_cagr*100:.2f}%)")
    print(f"Max Drawdown = {avg_maxdd*100:.2f}% (±{std_maxdd*100:.2f}%)")
    print(f"Win Rate     = {avg_wr*100:.2f}% (±{std_wr*100:.2f}%)")
    
    # Plot equity curves
    save_path = Path("data/results/equity_curves.png")
    plot_equity_curve(equity_curves, save_path, title="Forward Test Equity Curves")

if __name__ == "__main__":
    main()



# # kimchi_main.py
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import os
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import Tuple, Dict, List
# from collections import defaultdict
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# from utils.data import load_data, split_data
# from utils.metrics import calc_metrics
# from utils.technical_indicators import calculate_atr, calculate_adx
# from utils.config import TradingConfig
# from utils.funding_generator import create_funding_scenarios
# from visualization.plots import plot_sharpe_heatmap, plot_equity_curve

# def calculate_directional_volatility(data: pd.DataFrame, window: int = 30) -> Tuple[pd.Series, pd.Series]:
#     """상승장/하락장에서의 변동성을 별도로 계산"""
#     returns = data['binance_close'].pct_change()
    
#     up_returns = returns.copy()
#     up_returns[returns <= 0] = np.nan
    
#     down_returns = returns.copy()
#     down_returns[returns >= 0] = np.nan
    
#     up_vol = up_returns.rolling(window, min_periods=1).std() * np.sqrt(252)
#     down_vol = down_returns.rolling(window, min_periods=1).std() * np.sqrt(252)
    
#     return up_vol, down_vol

# def calculate_trend_strength(data: pd.DataFrame, window: int = 20) -> pd.Series:
#     """추세 강도 계산"""
#     atr = calculate_atr(data, window)
#     adx = calculate_adx(data, window)
#     return adx / (atr * 100)  # 0~1 사이 값으로 정규화

# def detect_extreme_market(data: pd.DataFrame, row_idx: int) -> bool:
#     """극단적 시장 상황 감지"""
#     if row_idx < 30:  # 충분한 데이터가 없는 경우
#         return False
        
#     current_data = data.iloc[row_idx]
#     hist_data = data.iloc[max(0, row_idx-30):row_idx]
    
#     conditions = [
#         # 1. 높은 변동성
#         current_data['vol_30d'] > 1.0,  # 100% 이상의 연간화 변동성
        
#         # 2. 급격한 가격 변화
#         abs(current_data['binance_close'] / data['binance_close'].iloc[row_idx-1] - 1) > 0.1,
        
#         # 3. 거래소간 큰 괴리
#         abs(current_data['upbit_close'] / current_data['binance_close'] - 1) > 0.05,
        
#         # 4. 최근 고점/저점 대비 큰 변화
#         current_data['binance_close'] < hist_data['binance_close'].min() * 0.8,
#         current_data['binance_close'] > hist_data['binance_close'].max() * 1.2
#     ]
    
#     return any(conditions)

# def dynamic_stop_loss(position_type: str, vol_30d: float, trend_strength: float,
#                      up_vol: float, down_vol: float, config: TradingConfig) -> float:
#     """더 보수적인 동적 손절 계산"""
#     base_stop = config.daily_stop_loss
    
#     # 1. 포지션 타입별 기본 조정
#     if position_type == 'short':
#         base_stop *= 0.7  # 더 타이트하게
#         vol_factor = up_vol / 0.4  # 상승 변동성 기준
#     else:  # long
#         vol_factor = down_vol / 0.4  # 하락 변동성 기준
    
#     # 2. 변동성 조정
#     vol_adjustment = 0.5 / max(vol_factor, 0.5)  
#     # 3. 추세 강도 조정
#     trend_adjustment = 1 + (trend_strength * 0.5)  
#     # 4. 극단적 상황 대비 최소 손절폭 설정
#     final_stop = base_stop * vol_adjustment * trend_adjustment
#     return max(final_stop, config.daily_stop_loss * 0.5)  # 최소 손절폭 보장

# def compute_position_size(capital: float, price: float, position_type: str,
#                         vol_30d: float, funding_rate: float, up_vol: float, 
#                         down_vol: float, trend_strength: float, 
#                         is_extreme_market: bool, config: TradingConfig) -> float:
#     """보수적인 포지션 사이즈 계산"""
#     # 1. 기본 position size 계산
#     max_pos = min(capital * config.leverage * config.position_size, 
#                  config.max_position_usd)
    
#     # 2. 변동성 기반 조정
#     if position_type == 'long':
#         vol_scale = 0.8 / (down_vol / 0.5)  
#     else:  # short
#         vol_scale = 0.8 / (up_vol / 0.5)   
    
#     max_pos *= np.clip(vol_scale, 0.2, 0.8)  
    
#     # 3. 포지션 타입별 기본 리스크 조정
#     if position_type == 'long':
#         max_pos *= 0.8  # 더 보수적으로
#         if funding_rate > 0:
#             max_pos *= np.exp(-7 * funding_rate)  # 펀딩비 영향 강화
#     else:  # short
#         max_pos *= 0.7  # 더 보수적으로
#         if funding_rate < 0:
#             max_pos *= np.exp(7 * funding_rate)   # 펀딩비 영향 강화
    
#     # 4. 추세 강도에 따른 조정
#     trend_scale = 1 + (trend_strength * 0.3)  
#     max_pos *= np.clip(trend_scale, 0.5, 1.2)
    
#     # 5. 극단적 시장 상황 대응
#     if is_extreme_market:
#         max_pos *= 0.3  
    
#     # 6. 최종 포지션 크기 제한
#     return min(max_pos / price, config.max_position_usd / price)

# def compute_daily_strategy_returns(df: pd.DataFrame, funding_df: pd.DataFrame, 
#                                X: float, Y: float, config: TradingConfig) -> pd.Series:
#     """리스크가 조정된 전략 수익률 계산"""
#     data = df.copy()
#     # 여기서는 기존처럼 pct_change() * 100 사용
#     data['upbit_pct_change'] = data['upbit_close'].pct_change() * 100
    
#     # 기본 지표 계산
#     data['vol_30d'] = data['binance_close'].pct_change().rolling(30).std() * np.sqrt(252)
#     data['up_vol'], data['down_vol'] = calculate_directional_volatility(data)
#     data['trend_strength'] = calculate_trend_strength(data)
    
#     signals = np.zeros(len(data))
#     position_sizes = np.zeros(len(data))
#     capital = config.initial_capital
    
#     # 수익률 저장용
#     price_returns = []
#     funding_returns = []
    
#     for i in range(1, len(data)):
#         # 현재 시장 상태 확인
#         is_extreme = detect_extreme_market(data, i)
#         current_time = data.index[i]
#         current_funding = funding_df.loc[current_time, 'funding_rate'] if current_time in funding_df.index else 0
        
#         # 새로운 시그널 생성
#         if signals[i-1] == 0:
#             # upbit_pct_change >= X일 때 롱
#             if data['upbit_pct_change'].iloc[i] >= X:
#                 signals[i] = 1
#                 position_sizes[i] = compute_position_size(
#                     capital=capital,
#                     price=data['binance_close'].iloc[i],
#                     position_type='long',
#                     vol_30d=data['vol_30d'].iloc[i],
#                     funding_rate=current_funding,
#                     up_vol=data['up_vol'].iloc[i],
#                     down_vol=data['down_vol'].iloc[i],
#                     trend_strength=data['trend_strength'].iloc[i],
#                     is_extreme_market=is_extreme,
#                     config=config
#                 )
#             # upbit_pct_change <= -Y일 때 숏
#             elif data['upbit_pct_change'].iloc[i] <= -Y:
#                 signals[i] = -1
#                 position_sizes[i] = compute_position_size(
#                     capital=capital,
#                     price=data['binance_close'].iloc[i],
#                     position_type='short',
#                     vol_30d=data['vol_30d'].iloc[i],
#                     funding_rate=current_funding,
#                     up_vol=data['up_vol'].iloc[i],
#                     down_vol=data['down_vol'].iloc[i],
#                     trend_strength=data['trend_strength'].iloc[i],
#                     is_extreme_market=is_extreme,
#                     config=config
#                 )
#         else:
#             # 기존 포지션 유지
#             signals[i] = signals[i-1]
#             position_sizes[i] = position_sizes[i-1]
            
#             # 동적 손절 체크
#             daily_return = data['binance_close'].pct_change().iloc[i]
#             stop_loss = dynamic_stop_loss(
#                 position_type='long' if signals[i] > 0 else 'short',
#                 vol_30d=data['vol_30d'].iloc[i],
#                 trend_strength=data['trend_strength'].iloc[i],
#                 up_vol=data['up_vol'].iloc[i],
#                 down_vol=data['down_vol'].iloc[i],
#                 config=config
#             )
#             # 손절 조건 충족 시 포지션 청산
#             if daily_return * signals[i] < -stop_loss:
#                 signals[i] = 0
#                 position_sizes[i] = 0
        
#         # 가격 변화로 인한 수익률 계산
#         price_return = signals[i-1] * position_sizes[i-1] * data['binance_close'].pct_change().iloc[i]
#         price_returns.append(price_return)
        
#         # 펀딩비 정산 시점일 경우 펀딩비 반영
#         if current_time in funding_df.index:
#             funding_return = -signals[i-1] * position_sizes[i-1] * current_funding
#             funding_returns.append(funding_return)
#         else:
#             funding_returns.append(0.0)
            
#         # 자본금 업데이트
#         capital = capital * (1 + price_return + funding_returns[-1])
    
#     # 일별 수익률로 변환
#     returns_df = pd.DataFrame({
#         'price_returns': price_returns,
#         'funding_returns': funding_returns
#     }, index=data.index[1:])
    
#     return (returns_df['price_returns'] + returns_df['funding_returns']).resample('D').sum().fillna(0)

# def parallel_monte_carlo_optimization(args: Tuple) -> Tuple:
#     """Monte Carlo 최적화를 위한 병렬 처리 함수"""
#     X, Y, price_df, funding_df, config = args
#     daily_rets = compute_daily_strategy_returns(price_df, funding_df, X, Y, config)
#     metrics = calc_metrics(daily_rets)
#     return (X, Y, metrics['Sharpe'])

# def process_monte_carlo_results(results: defaultdict, X_list: np.ndarray, 
#                               Y_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple]:
#     """Monte Carlo 시뮬레이션 결과 처리"""
#     results_matrix = np.zeros((len(X_list), len(Y_list)))
#     results_std = np.zeros((len(X_list), len(Y_list)))
#     best_sharpe = -np.inf
#     best_params = None
    
#     for i, X in enumerate(X_list):
#         for j, Y in enumerate(Y_list):
#             sharpe_values = results[(X, Y)]
#             avg_sharpe = np.mean(sharpe_values)
#             std_sharpe = np.std(sharpe_values)
            
#             results_matrix[i, j] = avg_sharpe
#             results_std[i, j] = std_sharpe
            
#             if avg_sharpe > best_sharpe:
#                 best_sharpe = avg_sharpe
#                 best_params = (X, Y)
    
#     return results_matrix, results_std, best_params

# def main():
#     # Configuration
#     config = TradingConfig(
#         initial_capital=100000,
#         leverage=3.0,
#         position_size=0.5,
#         daily_stop_loss=0.02,
#         max_position_usd=1000000
#     )
    
#     # Load data
#     price_df, _ = load_data("data/raw/")
#     print(f"\nTotal data range: {price_df.index.min()} ~ {price_df.index.max()}")
#     print(f"Total samples: {len(price_df):,} hourly data points")
    
#     # === 1) Split data (훈련 50% : 테스트 50%) ===
#     price_df_train, price_df_test = split_data(price_df, ratio=0.5) 
#     print(f"\nTrain period: {price_df_train.index.min()} ~ {price_df_train.index.max()}")
#     print(f"Test period: {price_df_test.index.min()} ~ {price_df_test.index.max()}")
#     print(f"Train set size: {len(price_df_train)}, Test set size: {len(price_df_test)}")
    
#     # Create funding scenarios
#     funding_scenarios_train = create_funding_scenarios(price_df_train, n_scenarios=5)
#     funding_scenarios_test = create_funding_scenarios(price_df_test, n_scenarios=5)
    
#     # Grid search parameters (범위 축소)
#     X_list = np.arange(0.01, 0.06, 0.005)  # 1% ~ 5% (각각 0.005 간격)
#     Y_list = np.arange(0.01, 0.06, 0.005)  # 1% ~ 5% (각각 0.005 간격)
    
#     # Set up parallel processing
#     n_cores = min(4, os.cpu_count())
#     print(f"\nUsing {n_cores} CPU cores for parallel processing")
    
#     # Create parameter scenarios
#     param_scenarios = [
#         (x, y, price_df_train, scenario, config)
#         for x in X_list
#         for y in Y_list
#         for scenario in funding_scenarios_train
#     ]
    
#     print(f"\nStarting Monte Carlo optimization with {len(param_scenarios)} total scenarios...")
#     results = defaultdict(list)
    
#     with ProcessPoolExecutor(max_workers=n_cores) as executor:
#         futures = [executor.submit(parallel_monte_carlo_optimization, p) 
#                   for p in param_scenarios]
        
#         for future in tqdm(as_completed(futures), total=len(futures)):
#             try:
#                 X, Y, sharpe = future.result()
#                 results[(X, Y)].append(sharpe)
#             except Exception as e:
#                 print(f"Error in optimization: {e}")
#                 continue
    
#     # Process results
#     results_matrix, results_std, best_params = process_monte_carlo_results(
#         results, X_list, Y_list)
    
#     # Backtest results
#     print("\n[Backtest Results]")
#     print(f"Best Parameters: X={best_params[0]*100:.2f}%, Y={best_params[1]*100:.2f}%")
#     best_i = np.where(X_list == best_params[0])[0][0]
#     best_j = np.where(Y_list == best_params[1])[0][0]
#     print(f"Average Sharpe Ratio: {results_matrix[best_i, best_j]:.4f}")
#     print(f"Sharpe Ratio Std: {results_std[best_i, best_j]:.4f}")
    
#     # Plot results
#     save_path = Path("data/results/kimchi_strategy_heatmap.png")
#     plot_sharpe_heatmap(results_matrix, X_list, Y_list, save_path, 
#                        title="Average Sharpe Ratio Across Scenarios")
    
#     save_path = Path("data/results/sharpe_std_heatmap.png")
#     plot_sharpe_heatmap(results_std, X_list, Y_list, save_path,
#                        title="Sharpe Ratio Standard Deviation Across Scenarios")
    
#     # Forward test
#     print("\nForward Testing Best Parameters")
#     best_X, best_Y = best_params
#     print(f"Using: X={best_X*100:.2f}%, Y={best_Y*100:.2f}%")
    
#     forward_metrics = []
#     equity_curves = []
    
#     for funding_df in tqdm(funding_scenarios_test, desc="Running forward tests"):
#         daily_rets = compute_daily_strategy_returns(
#             price_df_test, funding_df, best_X, best_Y, config)
#         metrics = calc_metrics(daily_rets)
#         forward_metrics.append(metrics)
#         equity_curves.append((1 + daily_rets).cumprod())
    
#     # Calculate and print average metrics
#     avg_metrics = {
#         'Sharpe': np.mean([m['Sharpe'] for m in forward_metrics]),
#         'CAGR': np.mean([m['CAGR'] for m in forward_metrics]),
#         'MaxDD': np.mean([m['MaxDD'] for m in forward_metrics]),
#         'WinRate': np.mean([m['WinRate'] for m in forward_metrics])
#     }
#     std_metrics = {
#         'Sharpe': np.std([m['Sharpe'] for m in forward_metrics]),
#         'CAGR': np.std([m['CAGR'] for m in forward_metrics]),
#         'MaxDD': np.std([m['MaxDD'] for m in forward_metrics]),
#         'WinRate': np.std([m['WinRate'] for m in forward_metrics])
#     }
    
#     print("\n[Forward Test Results (Average ± Std)]")
#     print(f"Sharpe Ratio = {avg_metrics['Sharpe']:.4f} (±{std_metrics['Sharpe']:.4f})")
#     print(f"CAGR        = {avg_metrics['CAGR']*100:.2f}% (±{std_metrics['CAGR']*100:.2f}%)")
#     print(f"Max Drawdown = {avg_metrics['MaxDD']*100:.2f}% (±{std_metrics['MaxDD']*100:.2f}%)")
#     print(f"Win Rate     = {avg_metrics['WinRate']*100:.2f}% (±{std_metrics['WinRate']*100:.2f}%)")
    
#     # Plot equity curves
#     save_path = Path("data/results/equity_curves.png")
#     plt.figure(figsize=(12, 6))
#     for curve in equity_curves:
#         plt.plot(curve.index, curve.values, alpha=0.2, color='blue')
#     plt.plot(equity_curves[0].index,
#             np.mean([curve.values for curve in equity_curves], axis=0),
#             'b-', linewidth=2, label='Average')
#     plt.title("Forward Test Equity Curves Across Scenarios")
#     plt.xlabel("Date")
#     plt.ylabel("Portfolio Value")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(save_path)
#     plt.close()

# if __name__ == "__main__":
#     main()
