# main.py
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_strategy import AdvancedKimchiStrategy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str = 'data/raw/') -> pd.DataFrame:
    """데이터 로드 및 전처리"""
    try:
        # 데이터 로드
        binance_df = pd.read_csv(f'{data_path}binance_perpetual.csv', index_col='timestamp', parse_dates=True)
        upbit_df   = pd.read_csv(f'{data_path}upbit_price.csv',       index_col='timestamp', parse_dates=True)
        
        # 데이터 병합 (inner join - 교집합만 남긴다고 가정)
        df = pd.DataFrame({
            'binance_close': binance_df['close'],
            'upbit_close':   upbit_df['close']
        })
        
        # Kimchi Premium 계산
        df['kimchi_premium'] = (df['upbit_close'] / df['binance_close'] - 1) * 100
        
        # Funding Rate 추가 (예시값, 실제로는 Binance API에서 가져와야 함)
        df['funding_rate'] = 0.01 * np.random.randn(len(df))  # 임시로 랜덤값 사용
        
        df.sort_index(inplace=True)
        logger.info(f"Data loaded. Shape={df.shape}. Range: {df.index.min()} ~ {df.index.max()}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_visualization(result_df: pd.DataFrame, report_df: pd.DataFrame, save_path: str = 'data/results/'):
    """결과 시각화"""
    try:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        if result_df.empty or report_df.empty:
            logger.warning("Either result_df or report_df is empty. Skipping visualization.")
            return
        
        # 1. 수익률 및 드로다운 차트
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(report_df.index, report_df['Cumulative_Return'], label='Cumulative Return')
        plt.title('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.fill_between(report_df.index, report_df['Drawdown'], 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}performance.png')
        plt.close()
        
        # 2. Regime 분석 히트맵
        regime_returns = pd.DataFrame({
            'Regime':  result_df['regime'],
            'Returns': result_df['final_returns']
        }, index=result_df.index)
        
        if not regime_returns.empty and isinstance(regime_returns.index, pd.DatetimeIndex):
            pivot_table = regime_returns.pivot_table(
                values='Returns',
                index=pd.Grouper(freq='M'),
                columns='Regime',
                aggfunc='sum'
            )
            if pivot_table.empty:
                logger.warning("Pivot table is empty. Skipping regime heatmap.")
            else:
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_table, annot=True, cmap='RdYlBu', center=0)
                plt.title('Monthly Returns by Regime')
                plt.savefig(f'{save_path}regime_analysis.png')
                plt.close()
        else:
            logger.warning("No valid regime_returns data or non-datetime index. Skipping heatmap.")
        
        # 3. 위험 메트릭 차트
        if 'Value_at_Risk' in report_df.columns and 'Expected_Shortfall' in report_df.columns:
            plt.figure(figsize=(15, 5))
            plt.plot(report_df.index, report_df['Value_at_Risk'], label='VaR (5%)', color='red')
            plt.plot(report_df.index, report_df['Expected_Shortfall'], label='Expected Shortfall', color='darkred')
            plt.title('Risk Metrics Over Time')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{save_path}risk_metrics.png')
            plt.close()
        else:
            logger.warning("Value_at_Risk or Expected_Shortfall not found in report_df. Skipping risk_metrics chart.")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise

def save_results(results: dict, filepath: str = 'data/results/backtest_results.json'):
    """결과를 JSON 파일로 저장"""
    try:
        # datetime 객체를 문자열로 변환
        serializable_results = {
            'train_period': [str(d) for d in results['train_period']],
            'test_period':  [str(d) for d in results['test_period']],
            'optimal_parameters': results['optimal_parameters'],
            'test_metrics': {}
        }
        
        # metrics 처리
        for key, value in results['test_metrics'].items():
            # numpy float -> 파이썬 float
            if isinstance(value, (np.float64, np.float32)):
                serializable_results['test_metrics'][key] = float(value)
            # numpy int -> 파이썬 int    
            elif isinstance(value, (np.int64, np.int32)):
                serializable_results['test_metrics'][key] = int(value)
            # dict (Regime_Statistics 등)
            elif isinstance(value, dict):
                sub_dict = {}
                for k2, v2 in value.items():
                    if isinstance(v2, (np.float64, np.float32)):
                        sub_dict[k2] = float(v2)
                    elif isinstance(v2, (np.int64, np.int32)):
                        sub_dict[k2] = int(v2)
                    else:
                        sub_dict[k2] = v2
                serializable_results['test_metrics'][key] = sub_dict
            else:
                serializable_results['test_metrics'][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"Results saved to {filepath}")
            
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def print_results(results: dict):
    """백테스트 결과 출력"""
    logger.info("\n=== Backtest Results ===")
    
    logger.info("\nPeriods:")
    logger.info(f"Training: {results['train_period'][0]} to {results['train_period'][1]}")
    logger.info(f"Testing: {results['test_period'][0]} to {results['test_period'][1]}")
    
    metrics = results['test_metrics']
    logger.info("\nPerformance Metrics:")
    logger.info(f"CAGR: {metrics['CAGR']*100:.2f}%")
    logger.info(f"Total Return: {metrics['Total_Return']*100:.2f}%")
    logger.info(f"Maximum Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
    logger.info(f"Sortino Ratio: {metrics['Sortino_Ratio']:.2f}")
    logger.info(f"Calmar Ratio: {metrics['Calmar_Ratio']:.2f}")
    
    logger.info("\nTrading Statistics:")
    logger.info(f"Total Trades: {metrics['Total_Trades']}")
    logger.info(f"Win Rate: {metrics['Win_Rate']*100:.2f}%")
    logger.info(f"Average Leverage: {metrics['Average_Leverage']:.2f}x")
    logger.info(f"Average Funding Cost: {metrics['Average_Funding_Cost']*100:.4f}%")

def main():
    try:
        # 결과 디렉토리 생성
        Path('data/results').mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading and preprocessing data...")
        df = load_and_preprocess_data()
        
        # 기간 설정
        train_period = ('2022-01-01', '2022-12-31')
        test_period  = ('2023-01-01', '2023-12-31')
        
        # 학습/테스트 데이터 분할
        train_data = df[train_period[0] : train_period[1]]
        test_data  = df[test_period[0]  : test_period[1]]
        
        logger.info(f"Train data shape: {train_data.shape}, range: {train_data.index.min()} ~ {train_data.index.max()}")
        logger.info(f"Test  data shape: {test_data.shape},  range: {test_data.index.min()} ~ {test_data.index.max()}")
        
        if test_data.empty:
            logger.warning("Test data is empty for the specified period. Backtest will produce empty results.")
        
        # 전략 인스턴스 생성
        strategy = AdvancedKimchiStrategy(
            n_states=3,
            leverage_limits=[1.0, 2.0, 3.0],
            stop_loss_limits=[0.01, 0.02, 0.03],
            take_profit_limits=[0.02, 0.03, 0.05],
            confidence_threshold=0.7,
            base_spread=0.0002  
        )
        
        logger.info("Running backtest with test data...")
        result_df, metrics = strategy.backtest(test_data)
        
        # 결과 리포트 생성
        logger.info("Generating report...")
        report_df = strategy.generate_report(result_df, metrics)
        
        # 결과 저장
        results = {
            'train_period': train_period,
            'test_period':  test_period,
            'optimal_parameters': {
                'n_states': strategy.n_states,
                'leverage_limits': strategy.leverage_limits,
                'stop_loss_limits': strategy.stop_loss_limits,
                'take_profit_limits': strategy.take_profit_limits
            },
            'test_metrics': metrics
        }
        
        save_results(results)
        print_results(results)
        
        # 시각화
        create_visualization(result_df, report_df)
        
        logger.info("\nBacktest completed successfully!")
        logger.info("Results have been saved to 'data/results/backtest_results.json'")
        logger.info("Visualizations have been saved to 'data/results/' directory")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
