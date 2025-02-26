import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from strategy import KimchiStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    """백테스팅 및 최적화를 수행하는 클래스"""
    
    def __init__(self, data_path: str = 'data/raw/'):
        """
        Args:
            data_path (str): 데이터 파일이 저장된 경로
        """
        self.data_path = data_path
        Path('data/results').mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """
        Binance와 Upbit 데이터를 로드하고 전처리
        
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        try:
            # 데이터 로드
            binance_df = pd.read_csv(f'{self.data_path}binance_perpetual.csv')
            upbit_df = pd.read_csv(f'{self.data_path}upbit_price.csv')
            
            # 타임스탬프를 datetime으로 변환
            binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'])
            upbit_df['timestamp'] = pd.to_datetime(upbit_df['timestamp'])
            
            # 데이터 병합
            merged_df = pd.merge(
                binance_df,
                upbit_df,
                on='timestamp',
                suffixes=('_binance', '_upbit')
            )
            
            # 컬럼명 변경
            merged_df = merged_df.rename(columns={
                'close_binance': 'binance_close',
                'close_upbit': 'upbit_close'
            })
            
            # 인덱스 설정
            merged_df = merged_df.set_index('timestamp').sort_index()
            
            # Funding Rate 계산 (8시간마다 0.01% 가정)
            merged_df['funding_rate'] = 0.0001  # 예시값
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def split_data(self, df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        데이터를 학습/테스트 세트로 분할
        
        Args:
            df (pd.DataFrame): 전체 데이터셋
            split_date (str): 분할 기준일 (YYYY-MM-DD)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (학습 데이터, 테스트 데이터)
        """
        train_data = df[df.index < split_date].copy()
        test_data = df[df.index >= split_date].copy()
        return train_data, test_data
        
    def optimize_strategy(self, 
                        train_data: pd.DataFrame,
                        long_range: Tuple[float, float, float] = (0.5, 5.0, 0.5),
                        short_range: Tuple[float, float, float] = (0.5, 5.0, 0.5)
                        ) -> Tuple[KimchiStrategy, pd.DataFrame]:
        """
        최적의 전략 파라미터 탐색
        
        Args:
            train_data (pd.DataFrame): 학습 데이터
            long_range (Tuple[float, float, float]): (시작, 끝, 간격) - Long 임계값
            short_range (Tuple[float, float, float]): (시작, 끝, 간격) - Short 임계값
            
        Returns:
            Tuple[KimchiStrategy, pd.DataFrame]: (최적화된 전략, 결과 데이터프레임)
        """
        # 테스트할 임계값 범위 생성
        long_thresholds = np.arange(long_range[0], long_range[1] + long_range[2], long_range[2])
        short_thresholds = np.arange(short_range[0], short_range[1] + short_range[2], short_range[2])
        
        # 기본 전략 인스턴스 생성
        strategy = KimchiStrategy(0, 0)  # 임시 값으로 초기화
        
        # 최적화 수행
        best_long, best_short, best_sharpe, results_df = strategy.optimize_parameters(
            train_data,
            long_thresholds,
            short_thresholds
        )
        
        # 최적화된 전략 생성
        optimized_strategy = KimchiStrategy(best_long, best_short)
        
        return optimized_strategy, results_df
        
    def create_heatmap(self, results_df: pd.DataFrame, save_path: str = 'data/results/heatmap.png'):
        """
        최적화 결과를 히트맵으로 시각화
        
        Args:
            results_df (pd.DataFrame): 최적화 결과
            save_path (str): 저장 경로
        """
        try:
            # 히트맵 데이터 준비
            pivot_table = results_df.pivot(
                index='Long_Threshold',
                columns='Short_Threshold',
                values='Sharpe_Ratio'
            )
            
            # 플롯 생성
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_table,
                annot=True,
                cmap='RdYlBu',
                center=0,
                fmt='.2f'
            )
            
            plt.title('Sharpe Ratio Heatmap')
            plt.xlabel('Short Threshold (%)')
            plt.ylabel('Long Threshold (%)')
            
            # 저장
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            raise
            
    def run_backtest(self, 
                    train_period: Tuple[str, str],
                    test_period: Tuple[str, str],
                    long_range: Tuple[float, float, float] = (0.5, 5.0, 0.5),
                    short_range: Tuple[float, float, float] = (0.5, 5.0, 0.5)
                    ) -> Dict:
        """
        전체 백테스트 프로세스 실행
        
        Args:
            train_period (Tuple[str, str]): (시작일, 종료일) - 학습 기간
            test_period (Tuple[str, str]): (시작일, 종료일) - 테스트 기간
            long_range (Tuple[float, float, float]): Long 임계값 범위
            short_range (Tuple[float, float, float]): Short 임계값 범위
            
        Returns:
            Dict: 백테스트 결과
        """
        try:
            # 데이터 로드
            df = self.load_data()
            
            # 학습/테스트 데이터 분할
            train_data = df[train_period[0]:train_period[1]]
            test_data = df[test_period[0]:test_period[1]]
            
            logger.info("Optimizing strategy parameters...")
            strategy, results_df = self.optimize_strategy(
                train_data,
                long_range,
                short_range
            )
            
            # 히트맵 생성
            self.create_heatmap(results_df)
            
            logger.info("Running backtest on test data...")
            test_data = strategy.generate_signals(test_data)
            _, test_metrics = strategy.backtest(test_data)
            
            # 결과 저장
            results = {
                'train_period': train_period,
                'test_period': test_period,
                'optimal_parameters': {
                    'long_threshold': strategy.long_threshold,
                    'short_threshold': strategy.short_threshold
                },
                'test_metrics': test_metrics
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest process: {str(e)}")
            raise