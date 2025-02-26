import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass

def create_realistic_funding_data(price_df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """
    Binance Perpetual Futures의 펀딩비 시나리오 생성
    
    Args:
        price_df: Upbit(BTC/KRW)와 Binance Futures(BTC/USDT) 가격 데이터
        seed: 랜덤 시드
    
    Returns:
        8시간 간격의 펀딩비 DataFrame
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 8hrs resampling (every 8hours)
    resampled_df = price_df.resample('8H').last()
    
    # Binance Perpetual Futures funding fee generate
    # general funding fee range: -0.1% ~ 0.1% (-0.001 ~ 0.001)
    n_periods = len(resampled_df)
    
    # basic funding fee generate (reflect mean revert)
    funding_rates = np.zeros(n_periods)
    mean_funding = 0.0001  # 약간의 롱 편향 가정
    mean_revert_speed = 0.7
    volatility = 0.0002
    
    # init funding fee
    funding_rates[0] = np.random.normal(mean_funding, volatility)
    
    # other funding fees (reflect mean revert)
    for i in range(1, n_periods):
        funding_rates[i] = (funding_rates[i-1] * (1 - mean_revert_speed) + 
                          mean_funding * mean_revert_speed + 
                          np.random.normal(0, volatility))
    
    # limit (-0.1% ~ 0.1%)
    funding_rates = np.clip(funding_rates, -0.001, 0.001)
    
    return pd.DataFrame({
        'funding_rate': funding_rates
    }, index=resampled_df.index)

def create_funding_scenarios(price_df: pd.DataFrame, n_scenarios: int = 10) -> List[pd.DataFrame]:
    """
    여러 개의 펀딩비 시나리오 생성
    
    Args:
        price_df: 가격 데이터
        n_scenarios: 생성할 시나리오 수
    
    Returns:
        펀딩비 시나리오 DataFrame 리스트
    """
    return [create_realistic_funding_data(price_df, seed=i) for i in range(n_scenarios)]