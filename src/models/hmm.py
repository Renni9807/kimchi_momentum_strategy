import numpy as np
from typing import Tuple, List
from sklearn.mixture import GaussianMixture

class LeverageHMM:
    """Hidden Markov Model for leverage regime detection"""
    
    def __init__(self, n_states: int = 3):
        """
        Initialize HMM model
        
        Args:
            n_states: Number of market regimes/states
        """
        self.n_states = n_states
        self.gmm = GaussianMixture(
            n_components=n_states,
            covariance_type='full',
            random_state=42
        )
        self.state_means_ = None
        self.state_covariances_ = None
        self.transition_matrix_ = None
        
    def _calc_features(self, upbit_returns: np.ndarray, window: int = 24) -> np.ndarray:
        """Calculate features for leverage decision"""
        features = np.column_stack([
            np.array([np.mean(upbit_returns[max(0, i-window):i]) 
                     for i in range(len(upbit_returns))]),  # Return trend
            np.array([np.std(upbit_returns[max(0, i-window):i]) 
                     for i in range(len(upbit_returns))]),  # Volatility
            np.array([np.sum(np.abs(upbit_returns[max(0, i-window):i]) > 1.0) / min(i+1, window)
                     for i in range(len(upbit_returns))])   # Large move frequency
        ])
        
        return features
        
    def fit(self, upbit_returns: np.ndarray, window: int = 24) -> 'LeverageHMM':
        """
        Fit HMM model
        
        Args:
            upbit_returns: Array of Upbit returns
            window: Rolling window size for feature calculation
            
        Returns:
            self
        """
        features = self._calc_features(upbit_returns, window)
        
        # Remove NaN rows
        features = features[~np.isnan(features).any(axis=1)]
        
        # Fit GMM
        self.gmm.fit(features)
        
        # Store parameters
        self.state_means_ = self.gmm.means_
        self.state_covariances_ = self.gmm.covariances_
        
        # Calculate transition matrix
        states = self.gmm.predict(features)
        transitions = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(states)-1):
            transitions[states[i], states[i+1]] += 1
            
        # Normalize transitions
        self.transition_matrix_ = transitions / transitions.sum(axis=1, keepdims=True)
        
        return self
    
    def predict_state(self, upbit_returns: np.ndarray, window: int = 24) -> int:
        """
        Predict current market regime
        
        Args:
            upbit_returns: Recent returns data
            window: Rolling window size
        """
        features = self._calc_features(upbit_returns[-window-1:], window)
        return self.gmm.predict(features[-1:])[-1]
    
    def get_state_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get fitted state parameters"""
        return self.state_means_, self.state_covariances_, self.transition_matrix_