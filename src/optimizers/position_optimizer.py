import numpy as np
import pandas as pd
from typing import Dict
from models.hmm import LeverageHMM
from models.particle import LeverageParticleFilter

class LeverageOptimizer:
    """Optimizer for dynamic leverage adjustment"""
    
    def __init__(self,
                 n_states: int = 3,
                 n_particles: int = 1000,
                 window: int = 24,
                 max_leverage: float = 3.0):
        """
        Initialize optimizer
        
        Args:
            n_states: Number of market regimes
            n_particles: Number of particles for filter
            window: Window size for feature calculation
            max_leverage: Maximum allowed leverage
        """
        self.hmm = LeverageHMM(n_states=n_states)
        self.particle_filter = LeverageParticleFilter(
            n_particles=n_particles,
            max_leverage=max_leverage
        )
        self.window = window
        self.current_state = None
        
    def fit(self, upbit_returns: np.ndarray) -> None:
        """
        Fit models to historical data
        
        Args:
            upbit_returns: Array of Upbit returns
        """
        # Fit HMM
        self.hmm.fit(upbit_returns, window=self.window)
        
        # Get initial state
        self.current_state = self.hmm.predict_state(
            upbit_returns[-self.window-1:], 
            window=self.window
        )
        
    def update(self, 
              current_return: float,
              window_returns: np.ndarray) -> Dict:
        """
        Update models and get new leverage recommendation
        
        Args:
            current_return: Latest strategy return
            window_returns: Recent return history
            
        Returns:
            Dict containing leverage recommendation and state info
        """
        # Update state estimate
        self.current_state = self.hmm.predict_state(
            window_returns,
            window=self.window
        )
        
        # Update particle filter
        self.particle_filter.update(
            returns=current_return,
            state=self.current_state
        )
        
        # Get leverage recommendation
        optimal_leverage = self.particle_filter.get_leverage_estimate()
        
        return {
            'optimal_leverage': optimal_leverage,
            'current_state': self.current_state,
            'state_info': self.get_state_info()
        }
    
    def get_state_info(self) -> Dict:
        """Get current state information"""
        state_means, state_covs, transitions = self.hmm.get_state_parameters()
        
        return {
            'current_state': self.current_state,
            'state_means': state_means,
            'state_covariances': state_covs,
            'transition_matrix': transitions,
            'state_interpretation': {
                0: 'Low Volatility - Aggressive',
                1: 'Medium Volatility - Moderate',
                2: 'High Volatility - Conservative'
            }
        }