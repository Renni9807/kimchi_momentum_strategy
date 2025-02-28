# Cryptocurrency Trading System with Kimchi Momentum Strategy

This repository contains a comprehensive system for cryptocurrency data collection, analysis, and algorithmic trading, implementing a "Kimchi Momentum" strategy that utilizes price movements in Korean markets to predict and trade on global exchanges.

## Project Overview

The system implements:
- Data collection from multiple exchanges (Binance, Upbit)
- Market volatility regime detection and analysis
- Dynamic position sizing and risk management
- Optimized trading parameters through Monte Carlo simulations
- Backtesting framework with performance analytics

## Key Components

### 1. Data Collection (DataFetcher)

The `DataFetcher` class optimizes data retrieval from cryptocurrency exchanges using:
- **Multithreaded execution** via ThreadPoolExecutor for parallel API requests
- **Exponential backoff** retry mechanism for handling API failures
- **Chunked data processing** to handle large time ranges efficiently
- **Progress tracking** with tqdm for monitoring data collection

```python
def fetch_parallel(self, exchange, symbol: str, chunks: List[Tuple[int, int]]) -> pd.DataFrame:
    all_candles = []
    
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        futures = {executor.submit(self._fetch_chunk, exchange, symbol, c[0], c[1]): c for c in chunks}
        # Process results as they complete
        for future in as_completed(futures):
            # Result processing...
```

### 2. Kimchi Momentum Strategy

The strategy captures momentum from Korean markets (Upbit) to execute trades on global markets (Binance) with:
- **Directional volatility analysis** that separately calculates volatility for uptrends and downtrends
- **Dynamic risk management** with state-dependent position sizing
- **Adaptive stop-loss mechanisms** based on market conditions
- **Monte Carlo optimization** using parallel processing to find optimal X and Y parameters

```python
def compute_position_size(capital: float, price: float, position_type: str,
                        vol_30d: float, funding_rate: float, up_vol: float, 
                        down_vol: float, trend_strength: float, 
                        is_extreme_market: bool, config: TradingConfig) -> float:
    # Dynamic position sizing based on market conditions
    # ...
```

### 3. Technical Indicators and Market Analysis

The system employs various technical indicators and analysis methods:
- **Average True Range (ATR)** for measuring volatility
- **Average Directional Index (ADX)** for trend strength analysis
- **Trend strength calculation** to detect market regime changes
- **Extreme market detection** to manage risk during unusual market conditions

## Performance Considerations

- The strategy simulation is optimized with multiprocessing for Monte Carlo analysis
- Data collection utilizes multithreading for efficient API interactions
- The current implementation balances computational efficiency with strategy robustness

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- matplotlib, seaborn for data visualization
- ccxt for exchange API connections
- concurrent.futures for parallel processing
- tqdm for progress tracking
- sklearn for machine learning models (for future enhancements)

## Advanced Visualization

The system includes comprehensive visualization tools for strategy analysis and performance evaluation:

### 1. Performance Heatmaps
```python
def plot_sharpe_heatmap(results_matrix: np.ndarray, 
                       X_list: Union[List[float], np.ndarray], 
                       Y_list: Union[List[float], np.ndarray], 
                       save_path: Union[str, Path],
                       title: Optional[str] = None) -> None:
    # Creates enhanced heatmap showing Sharpe ratio across parameter combinations
    # Highlights optimal parameters and includes top 5 parameter combinations
```

### 2. Equity Curve Analysis
```python
def plot_equity_curve(equity_curves: List[pd.Series], 
                     save_path: Union[str, Path],
                     title: Optional[str] = None,
                     show_percentiles: bool = True) -> None:
    # Plots multiple equity curves with percentile bands
    # Shows strategy performance variability across different scenarios
```

### 3. Returns Distribution
```python
def plot_returns_distribution(returns: pd.Series,
                            save_path: Union[str, Path],
                            title: Optional[str] = None) -> None:
    # Visualizes the distribution of strategy returns
    # Compares actual returns to normal distribution
```

### 4. Drawdown Analysis
```python
def plot_drawdown_periods(equity_curve: pd.Series,
                         save_path: Union[str, Path],
                         title: Optional[str] = None) -> None:
    # Analyzes and visualizes drawdown periods
    # Shows both equity curve and drawdown percentage
```

## Technical Indicators and Analysis Tools

The system includes implementations of various technical indicators and analysis tools:

### Technical Indicators
- Average True Range (ATR) for volatility measurement
- Average Directional Index (ADX) for trend strength analysis
- Directional volatility analysis (separate metrics for up/down moves)
- Trend strength calculation combining multiple factors

### Visualization Tools
The system provides comprehensive visualization capabilities to analyze strategy performance:
- Parameter optimization heatmaps with best parameter highlighting
- Multi-scenario equity curves with percentile bands
- Return distribution analysis with normal distribution comparison
- Drawdown period visualization and analysis

## Getting Started

1. Setup your API keys in configuration
2. Run data collection to gather historical data:
   ```
   python src/data_fetcher.py
   ```
3. Execute backtests to identify optimal parameters:
   ```
   python src/kimchi_main.py
   ```
4. Analyze results using the visualization tools:
   ```
   python src/analyze_results.py
   ```
5. Deploy the strategy with appropriate risk settings

## References

- Future implementation of auxiliary particle filtering could be based on "Simulation-based sequential analysis of Markov switching stochastic volatility models" by Carvalho and Lopes
- The dynamic position sizing methodology is custom-developed for cryptocurrency markets with high volatility
