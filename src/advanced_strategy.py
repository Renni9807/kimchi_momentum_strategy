# advanced_strategy.py
import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Tuple, Dict, List, Optional
import logging
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketImpactCalculator:
    """주문 크기에 따른 시장 충격 계산"""
    def __init__(self, base_spread: float = 0.0002):
        self.base_spread = base_spread
        
    def calculate_slippage(self, order_size: float, market_depth: float) -> float:
        """
        주문 크기와 시장 깊이에 따른 슬리피지 계산
        
        Args:
            order_size: 주문 크기 (BTC)
            market_depth: 주문창의 깊이 (BTC)
        """
        impact = self.base_spread * (1 + (order_size / market_depth) ** 0.5)
        return min(impact, 0.003)  # 최대 30bp로 제한

class LiquidationCalculator:
    """청산 가격 및 위험 계산"""
    def __init__(self, maintenance_margin: float = 0.004):
        self.maintenance_margin = maintenance_margin
        
    def calculate_liquidation_price(self, 
                                  entry_price: float, 
                                  position_size: float,
                                  leverage: float,
                                  is_long: bool) -> float:
        """청산 가격 계산"""
        margin_requirement = entry_price * abs(position_size) * self.maintenance_margin
        if is_long:
            return entry_price * (1 - 1/leverage + self.maintenance_margin)
        else:
            return entry_price * (1 + 1/leverage - self.maintenance_margin)
            
    def is_liquidated(self, 
                     current_price: float,
                     liquidation_price: float,
                     is_long: bool) -> bool:
        """현재 가격이 청산 가격을 trigger했는지 확인"""
        if is_long:
            return current_price <= liquidation_price
        return current_price >= liquidation_price

class FundingRateManager:
    """Funding Rate 관리 및 비용 계산"""
    def __init__(self, funding_interval: int = 8):
        self.funding_interval = funding_interval
        
    def calculate_funding_cost(self, 
                             position_size: float,
                             funding_rate: float,
                             hours_held: float) -> float:
        """
        Funding 비용 계산
        
        Args:
            position_size: 포지션 크기
            funding_rate: 현재 funding rate
            hours_held: 포지션 보유 시간
        """
        num_funding_events = hours_held / self.funding_interval
        total_funding_cost = position_size * funding_rate * num_funding_events
        return total_funding_cost

class ParticleFilter:
    """Particle Filter for State Estimation"""
    def __init__(self, n_particles: int = 1000):
        self.n_particles = n_particles
        self.particles = None
        self.weights = None
        
    def initialize(self, initial_state: float, noise_std: float):
        self.particles = np.random.normal(initial_state, noise_std, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def predict(self, process_noise_std: float):
        self.particles = self.particles + np.random.normal(0, process_noise_std, self.n_particles)
        
    def update(self, measurement: float, measurement_noise_std: float):
        likelihood = norm.pdf(measurement, self.particles, measurement_noise_std)
        self.weights *= likelihood
        self.weights /= np.sum(self.weights)  # Normalize
        
    def resample(self):
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        indices = np.searchsorted(cumsum, np.random.random(self.n_particles))
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def estimate(self) -> float:
        return np.average(self.particles, weights=self.weights)


class AdvancedKimchiStrategy:
    """Advanced Kimchi Premium Strategy with HMM and Particle Filter"""
    
    def __init__(self,
                 n_states: int = 3,
                 leverage_limits: List[float] = [1.0, 2.0, 3.0],
                 stop_loss_limits: List[float] = [0.01, 0.02, 0.03],
                 take_profit_limits: List[float] = [0.02, 0.03, 0.05],
                 confidence_threshold: float = 0.7,
                 binance_fee: float = 0.0004,
                 upbit_fee: float = 0.0005,
                 base_spread: float = 0.0002,  # Added base_spread parameter
                 margin_mode: str = "cross"):  # cross or isolated
        
        self.n_states = n_states
        self.leverage_limits = leverage_limits
        self.stop_loss_limits = stop_loss_limits
        self.take_profit_limits = take_profit_limits
        self.confidence_threshold = confidence_threshold
        self.binance_fee = binance_fee
        self.upbit_fee = upbit_fee
        self.margin_mode = margin_mode
        self.base_spread = base_spread  # Store base_spread
        
        # Initialize components
        self.hmm = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
        self.particle_filter = ParticleFilter()
        self.market_impact = MarketImpactCalculator(base_spread=base_spread)
        self.liquidation_calc = LiquidationCalculator()
        self.funding_manager = FundingRateManager()
        
        # Initialize scalers
        self.scaler = StandardScaler()
        
        # We'll define which columns we plan to scale
        self.feature_columns = [
            'premium', 'premium_std', 'premium_zscore',
            'vol_ratio', 'vol_diff', 'funding_cumsum'
        ]
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Feature engineering for HMM"""
        # Start with an empty DataFrame with the expected columns
        features = pd.DataFrame(index=df.index, columns=self.feature_columns)
        
        # Kimchi Premium features
        features['premium'] = df['kimchi_premium']
        features['premium_std'] = df['kimchi_premium'].rolling(24).std()
        features['premium_zscore'] = (
            features['premium'] - features['premium'].rolling(24).mean()
        ) / features['premium_std']
        
        # Volatility features
        features['vol_ratio'] = df['binance_volatility'] / df['upbit_volatility']
        features['vol_diff'] = df['binance_volatility'] - df['upbit_volatility']
        
        # Funding rate features (with default value of 0 if not present)
        if 'funding_rate' in df.columns:
            features['funding_cumsum'] = df['funding_rate'].rolling(24).sum()
        else:
            features['funding_cumsum'] = 0.0
        
        # Fill NaN values with 0
        features = features.fillna(0)
        
        # Scale features
        if len(features) > 0:
            scaled_features = self.scaler.fit_transform(features)
            return scaled_features
        return np.array([])
        
    def fit_hmm(self, df: pd.DataFrame):
        """Train HMM on historical data"""
        if df.empty:
            logger.warning("Training data is empty! Skipping HMM fit.")
            return
        
        # Calculate volatilities first
        df = df.copy()
        df['binance_volatility'] = df['binance_close'].pct_change().rolling(24).std()
        df['upbit_volatility']   = df['upbit_close'].pct_change().rolling(24).std()
        
        # Now prepare features
        features = self.prepare_features(df)
        
        # Fit the HMM if we have enough data
        if len(features) == 0:
            logger.warning("No valid features found for HMM training (features=0). Skipping HMM fit.")
            return
        
        try:
            self.hmm.fit(features)
            logger.info("HMM fit successful.")
        except Exception as e:
            logger.warning(f"HMM fit failed with exception: {e}")
        
    def get_regime_probabilities(self, current_features: pd.DataFrame) -> np.ndarray:
        """Get probability distribution over regimes"""
        feature_array = current_features.values
        scaled_features = self.scaler.transform(feature_array)
        
        # If HMM wasn't fitted (or failed), fallback to uniform distribution
        try:
            return self.hmm.predict_proba(scaled_features)[0]
        except:
            n = self.n_states
            return np.ones(n) / n
        
    def calculate_optimal_position(self,
                                 kimchi_premium: float,
                                 volatility: float,
                                 regime_probs: np.ndarray,
                                 particle_estimate: float,
                                 market_depth: float,
                                 funding_rate: float,
                                 current_price: float) -> Tuple[float, float, float, float]:
        """
        Calculate optimal position size and risk parameters
        
        Returns:
            (position_size, stop_loss, take_profit, liquidation_price)
        """
        regime = np.argmax(regime_probs)
        regime_confidence = regime_probs[regime]
        
        position_size = 0.0
        liquidation_price = 0.0
        
        if regime_confidence > self.confidence_threshold and volatility > 0:
            # Base position size
            position_size = np.clip(abs(kimchi_premium) / volatility, 0, 1)
            
            # Adjust for funding rate
            if abs(funding_rate) > 0.01:
                position_size *= 0.5
                
            # Adjust for market depth
            estimated_slippage = self.market_impact.calculate_slippage(position_size, market_depth)
            position_size *= (1 - estimated_slippage)
            
            # Apply leverage limit based on regime and volatility
            max_leverage = min(
                self.leverage_limits[regime],
                1 / (volatility * 2) if volatility != 0 else 5.0,
                5.0  # hard cap
            )
            
            position_size *= max_leverage
            position_size *= np.sign(kimchi_premium)  # long if premium>0, short if premium<0
            
            # Calculate liquidation price
            is_long = position_size > 0
            liquidation_price = self.liquidation_calc.calculate_liquidation_price(
                current_price,
                abs(position_size),
                max_leverage,
                is_long
            )
            
        return (
            position_size,
            self.stop_loss_limits[regime],
            self.take_profit_limits[regime],
            liquidation_price
        )
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with dynamic position sizing"""
        try:
            if df.empty:
                logger.warning("DataFrame is empty in generate_signals. Returning as is.")
                return df
            
            logger.info("Starting signal generation...")
            result_df = df.copy()
            
            # Calculate volatilities first
            logger.info("Calculating volatilities...")
            result_df['binance_volatility'] = result_df['binance_close'].pct_change().rolling(24).std()
            result_df['upbit_volatility']   = result_df['upbit_close'].pct_change().rolling(24).std()
            
            # Prepare features for entire dataset
            logger.info("Calculating features for entire dataset...")
            features = pd.DataFrame(index=result_df.index, columns=self.feature_columns)
            features['premium']       = result_df['kimchi_premium']
            features['premium_std']   = result_df['kimchi_premium'].rolling(24).std()
            features['premium_zscore'] = (
                features['premium'] - features['premium'].rolling(24).mean()
            ) / features['premium_std']
            features['vol_ratio']     = result_df['binance_volatility'] / result_df['upbit_volatility']
            features['vol_diff']      = result_df['binance_volatility'] - result_df['upbit_volatility']
            
            if 'funding_rate' in result_df.columns:
                features['funding_cumsum'] = result_df['funding_rate'].rolling(24).sum()
            else:
                features['funding_cumsum'] = 0.0
            
            # Fill NaN
            features = features.fillna(0)
            
            # Fit scaler once
            logger.info("Fitting scaler on entire dataset features...")
            if not features.empty:
                self.scaler.fit(features)
            else:
                logger.warning("Feature DataFrame is empty. No signals generated.")
                result_df['position']            = 0.0
                result_df['stop_loss']          = 0.0
                result_df['take_profit']        = 0.0
                result_df['liquidation_price']  = 0.0
                result_df['regime']             = 0
                result_df['regime_confidence']  = 0.0
                return result_df
            
            signals = []
            total_rows = len(result_df)
            logger.info("Processing each time step for signals...")
            for i in range(total_rows):
                # Periodic progress log
                if i % 1000 == 0:
                    logger.info(f"Processing row {i}/{total_rows}")
                
                try:
                    current_features = features.iloc[i:i+1]
                    
                    # Regime probabilities
                    regime_probs = self.get_regime_probabilities(current_features)
                    
                    # Calculate optimal position
                    position, sl, tp, liq_price = self.calculate_optimal_position(
                        kimchi_premium = result_df['kimchi_premium'].iloc[i],
                        volatility     = result_df['binance_volatility'].iloc[i] if not pd.isna(result_df['binance_volatility'].iloc[i]) else 0,
                        regime_probs   = regime_probs,
                        particle_estimate = 0,  # ignoring for now
                        market_depth   = 10.0,
                        funding_rate   = result_df['funding_rate'].iloc[i],
                        current_price  = result_df['binance_close'].iloc[i]
                    )
                    
                    signals.append({
                        'position': position,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'liquidation_price': liq_price,
                        'regime': np.argmax(regime_probs),
                        'regime_confidence': np.max(regime_probs)
                    })
                
                except Exception as e:
                    logger.warning(f"Error processing row {i}: {str(e)}")
                    signals.append({
                        'position': 0.0,
                        'stop_loss': 0.0,
                        'take_profit': 0.0,
                        'liquidation_price': 0.0,
                        'regime': 0,
                        'regime_confidence': 0.0
                    })
            
            # Merge signals into result_df
            logger.info("Converting signals to DataFrame...")
            signals_df = pd.DataFrame(signals, index=result_df.index)
            result_df = pd.concat([result_df, signals_df], axis=1)
            
            return result_df
                
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
            
    def calculate_vectorized_pnl(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized version of PnL calculation"""
        if df.empty:
            return pd.Series([], dtype=float)
        
        pnl = df['strategy_returns'].fillna(0)
        
        # Apply funding costs
        if 'funding_rate' in df.columns:
            # 8h funding, so approximate hours_held by 8/24?
            # This is a simplification
            funding_costs = (df['position'] * df['funding_rate'] * (8/24)).fillna(0)
            pnl -= funding_costs
        
        # Apply transaction costs
        position_changes = df['position_change'].fillna(0)
        transaction_mask = position_changes != 0
        leverage = df['position'].abs().fillna(0)
        
        slippage = np.minimum(
            self.base_spread * (1 + (abs(position_changes) / 10.0) ** 0.5),
            0.003
        )
        total_costs = transaction_mask * (self.binance_fee + self.upbit_fee + slippage) * leverage
        
        pnl -= total_costs
        
        return pnl
    
    def calculate_metrics(self, df: pd.DataFrame, initial_capital: float) -> Dict:
        """Calculate performance metrics"""
        if df.empty:
            logger.warning("Result DataFrame is empty. All metrics set to 0 or NaN.")
            return {
                'CAGR': 0,
                'Max_Drawdown': 0,
                'Sharpe_Ratio': 0,
                'Sortino_Ratio': 0,
                'Calmar_Ratio': 0,
                'Total_Return': 0,
                'Annual_Volatility': 0,
                'Total_Trades': 0,
                'Win_Rate': 0,
                'Average_Leverage': 0,
                'Average_Funding_Cost': 0,
                'Regime_Statistics': {}
            }
        
        # Calculate basic return statistics
        total_days = (df.index[-1] - df.index[0]).days
        total_years = total_days / 365.0 if total_days > 0 else 1.0
        
        final_value = float(df['cumulative_value'].iloc[-1])
        cagr = float((final_value / initial_capital) ** (1 / total_years) - 1) if final_value>0 else -1
        
        max_drawdown = float(df['drawdown'].max())
        
        daily_returns = df['final_returns'].resample('D').sum()
        annual_volatility = float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
        sharpe_ratio = float((cagr - 0) / annual_volatility) if annual_volatility != 0 else 0
        
        # Calculate additional metrics
        total_trades = int(df['position_change'].ne(0).sum())
        win_rate = float((df['final_returns'] > 0).mean()) if len(df) > 0 else 0
        
        avg_leverage = float(abs(df['position']).mean()) if len(df) > 0 else 0
        
        # Risk-adjusted metrics
        negative_returns = daily_returns[daily_returns < 0]
        neg_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = float(cagr / neg_std) if neg_std != 0 else 0
        calmar_ratio = float(abs(cagr / max_drawdown)) if max_drawdown != 0 else 0
        
        # Average funding cost
        avg_funding_cost = float(
            df['final_returns'].sub(
                df['strategy_returns'],
                fill_value=0
            ).mean()
        )
        
        # Regime statistics
        regime_stats = {}
        for regime in df['regime'].dropna().unique():
            regime_data = df[df['regime'] == regime]
            regime_stats[f'regime_{regime}_returns_mean']   = float(regime_data['final_returns'].mean())
            regime_stats[f'regime_{regime}_returns_std']    = float(regime_data['final_returns'].std())
            regime_stats[f'regime_{regime}_position_mean']  = float(regime_data['position'].mean())
            regime_stats[f'regime_{regime}_count']          = int(len(regime_data))
        
        return {
            'CAGR': cagr,
            'Max_Drawdown': max_drawdown,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Total_Return': float((final_value / initial_capital) - 1),
            'Annual_Volatility': annual_volatility,
            'Total_Trades': total_trades,
            'Win_Rate': win_rate,
            'Average_Leverage': avg_leverage,
            'Average_Funding_Cost': avg_funding_cost,
            'Regime_Statistics': regime_stats
        }
            
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Tuple[pd.DataFrame, Dict]:
        """Run backtest with comprehensive analysis"""
        try:
            logger.info("Starting backtest...")
            
            # Copy data
            result_df = df.copy()
            
            if result_df.empty:
                logger.warning("Test DataFrame is empty. Returning empty result + zero metrics.")
                empty_metrics = {
                    'CAGR': 0, 'Max_Drawdown': 0, 'Sharpe_Ratio': 0, 'Sortino_Ratio': 0, 'Calmar_Ratio': 0,
                    'Total_Return': 0, 'Annual_Volatility': 0, 'Total_Trades': 0, 'Win_Rate': 0,
                    'Average_Leverage': 0, 'Average_Funding_Cost': 0, 'Regime_Statistics': {}
                }
                return result_df, empty_metrics
            
            # Train HMM on the first portion of data (1/3 of test data, or some different logic)
            train_size = len(result_df) // 3
            logger.info(f"Training HMM with {train_size} samples (from test_data).")
            
            # Fit HMM (it will skip if not enough data)
            self.fit_hmm(result_df.head(train_size))
            
            # Generate trading signals
            logger.info("Generating trading signals...")
            result_df = self.generate_signals(result_df)
            
            logger.info("Calculating returns (binance_returns, strategy_returns, etc.)...")
            # Calculate returns
            result_df['binance_returns']  = result_df['binance_close'].pct_change()
            result_df['position_change']  = result_df['position'].diff()
            result_df['strategy_returns'] = result_df['position'].shift(1) * result_df['binance_returns']
            
            logger.info("Applying risk management (vectorized_pnl)...")
            # Vectorize calculations
            result_df['final_returns'] = self.calculate_vectorized_pnl(result_df)
            
            logger.info("Calculating cumulative returns and drawdown...")
            result_df['cumulative_returns'] = (1 + result_df['final_returns']).cumprod()
            result_df['cumulative_value']   = initial_capital * result_df['cumulative_returns']
            result_df['drawdown'] = 1 - result_df['cumulative_value'] / result_df['cumulative_value'].expanding().max()
            
            logger.info("Calculating performance metrics...")
            metrics = self.calculate_metrics(result_df, initial_capital)
            
            logger.info("Backtest completed successfully.")
            return result_df, metrics
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise


    def generate_report(self, df: pd.DataFrame, metrics: Dict) -> pd.DataFrame:
        """Generate detailed performance report"""
        try:
            if df.empty:
                logger.warning("DataFrame is empty in generate_report. Returning empty report.")
                return pd.DataFrame()
            
            report_df = pd.DataFrame()
            
            # Daily statistics
            daily_returns = df['final_returns'].resample('D').sum()
            report_df['Daily_Return'] = daily_returns
            report_df['Cumulative_Return'] = (1 + daily_returns).cumprod()
            
            # Volatility analysis
            report_df['Rolling_Volatility'] = daily_returns.rolling(30).std() * np.sqrt(252)
            report_df['Rolling_Sharpe'] = (
                daily_returns.rolling(30).mean() * 252 /
                (daily_returns.rolling(30).std() * np.sqrt(252))
            )
            
            # Drawdown analysis
            report_df['Drawdown'] = 1 - report_df['Cumulative_Return'] / report_df['Cumulative_Return'].expanding().max()
            
            # Regime analysis
            if 'regime' in df.columns:
                report_df['Regime'] = df['regime'].resample('D').last()
            if 'regime_confidence' in df.columns:
                report_df['Regime_Confidence'] = df['regime_confidence'].resample('D').mean()
            
            # Position analysis
            if 'position' in df.columns:
                report_df['Average_Position'] = df['position'].resample('D').mean()
            if 'position_change' in df.columns:
                report_df['Position_Changes'] = df['position_change'].resample('D').count()
            
            # Risk metrics
            if len(daily_returns) > 30:
                report_df['Value_at_Risk'] = daily_returns.rolling(30).quantile(0.05)
                report_df['Expected_Shortfall'] = daily_returns[
                    daily_returns < report_df['Value_at_Risk']
                ].rolling(30).mean()
            else:
                report_df['Value_at_Risk'] = np.nan
                report_df['Expected_Shortfall'] = np.nan
            
            return report_df
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
