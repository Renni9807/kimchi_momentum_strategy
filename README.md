.
├── kimchi_main.py # Main script for backtesting & forward testing
├── utils
│ ├── data.py # Data loading and splitting
│ ├── funding_generator.py # Monte Carlo funding-rate scenario generation
│ ├── metrics.py # Performance metrics (Sharpe, CAGR, etc.)
│ ├── config.py # Trading configuration (leverage, stop-loss, etc.)
│ └── technical_indicators.py # Technical indicators (ATR, ADX, etc.)
├── visualization
│ └── plots.py # Visualization functions (heatmaps, equity curves)
├── data
│ ├── raw # Directory for raw CSV data
│ └── results # Directory for backtest/plots output
└── README.md # This document

File Descriptions

1. kimchi_main.py
   Main execution script performing:

Data loading (load_data) and train/test split (split_data)
Funding scenarios generation (create_funding_scenarios)
Parameter grid definitions for (X, Y)
Parallelized backtesting (ProcessPoolExecutor) over multiple scenarios
Optimal parameter selection based on Sharpe Ratio
Forward testing and plotting (heatmaps and equity curves)
It contains the main() function. Run via:

python kimchi_main.py
Key function:

compute_daily_strategy_returns(): Generates signals based on X and Y thresholds and calculates the daily returns. 2. utils/data.py
Handles data input/output.
Key functions:
load_data(data_path):
Reads CSV files (upbit_price.csv, binance_perpetual.csv, binance_funding.csv) from data_path, merges them into a single price_df, and returns (price_df, funding_df).
split_data(df, ratio):
Splits the input DataFrame into train and test sets by a given ratio. 3. utils/funding_generator.py
Responsible for Monte Carlo funding-rate scenarios.
Key functions:
create_realistic_funding_data(price_df, seed):
Creates a single funding-rate series, resampled to 8-hour intervals, using a mean-reverting process within a range of -0.1% to +0.1%.
create_funding_scenarios(price_df, n_scenarios):
Generates multiple (n_scenarios) funding scenarios and returns them as a list of DataFrames. 4. utils/metrics.py
Computes performance metrics for backtesting.
Key function:
calc_metrics(daily_ret):
Given a daily returns series, calculates:
Sharpe Ratio
CAGR
Sortino Ratio
Maximum Drawdown
Win Rate 5. utils/technical_indicators.py
Contains logic for technical indicators such as ATR and ADX.
Key functions:
calculate_atr(data, window): Computes the Average True Range (ATR).
calculate_adx(data, window): Computes the Average Directional Movement Index (ADX). 6. utils/config.py
Defines a TradingConfig dataclass that bundles parameters like:
initial_capital, leverage, daily_stop_loss
maker_fee, taker_fee, slippage
Additional technical indicator settings (RSI, Bollinger Bands, MACD).
By adjusting this configuration object, you can modify the risk and fee settings globally.

7. visualization/plots.py
   Provides visualization functions for backtest results:
   plot_sharpe_heatmap(results_matrix, X_list, Y_list, save_path, ...):
   Creates a Seaborn heatmap of Sharpe Ratios, marking the best parameter combination with a star, and shows the top 5 parameter sets.
   plot_equity_curve(equity_curves, save_path, ...):
   Plots multiple equity curves from different funding scenarios on the same chart, optionally highlighting percentile bands.
   Additional utility plots (e.g., drawdown analysis, distribution of returns).
   How to Use
   Data Preparation

Place CSV files named upbit_price.csv, binance_perpetual.csv, and binance_funding.csv in the data/raw/ folder.
Make sure each CSV includes a timestamp column (parsed as DateTime) and a close column.
The load_data function reads these and merges them into a single DataFrame.
Running the Strategy

In a Python 3.8+ environment, install necessary packages (numpy, pandas, matplotlib, seaborn, tqdm).
Run:
python kimchi_main.py
The console will display the train/test results, best parameters, Sharpe Ratio, etc.
Result plots (heatmap, equity curves) will be saved under data/results/.
Modifying Parameters

To change the Monte Carlo scenarios (number of funding scenarios), update:

create_funding_scenarios(price_df_train, n_scenarios=5)
To adjust (X, Y) search ranges, modify:

X_list = np.arange(0.01, 0.06, 0.01)
Y_list = np.arange(0.01, 0.06, 0.01)
Or change the training/test split ratio in:

Dynamic leverage handing

Implement dynamic leverage adjustments using Hidden Markov Models (HMM) and Particle Filters to better respond to changing market conditions and optimize risk management.
