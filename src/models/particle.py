import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class Particle:
    """Particle class for leverage estimation"""
    leverage: float
    weight: float = 1.0

class LeverageParticleFilter:
    """Particle Filter for leverage optimization"""
    
    def __init__(self, n_particles: int = 1000,
                 max_leverage: float = 3.0):
        """
        Initialize Particle Filter
        
        Args:
            n_particles: Number of particles
            max_leverage: Maximum allowed leverage
        """
        self.n_particles = n_particles
        self.max_leverage = max_leverage
        self.particles = self._initialize_particles()
        
    def _initialize_particles(self) -> List[Particle]:
        """Initialize particles with random leverage values"""
        return [
            Particle(
                leverage=np.random.uniform(0, self.max_leverage),
                weight=1.0/self.n_particles
            )
            for _ in range(self.n_particles)
        ]
    
    def _resample(self) -> None:
        """Resample particles based on weights"""
        weights = np.array([p.weight for p in self.particles])
        weights /= weights.sum()
        
        # Systematic resampling
        positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        cumulative_sum = np.cumsum(weights)
        
        new_particles = []
        i = 0
        for position in positions:
            while cumulative_sum[i] < position:
                i += 1
            new_particles.append(Particle(
                leverage=self.particles[i].leverage
            ))
        
        self.particles = new_particles
        
    def _add_noise(self, noise_scale: float = 0.1) -> None:
        """Add noise to particles for exploration"""
        for p in self.particles:
            p.leverage += np.random.normal(0, noise_scale)
            p.leverage = np.clip(p.leverage, 0, self.max_leverage)
        
    def update(self, 
              returns: float,
              state: int,
              risk_free_rate: float = 0.02) -> None:
        """
        Update particle weights based on returns
        
        Args:
            returns: Strategy returns
            state: Current market regime
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        daily_rf = risk_free_rate / 252  # Daily risk-free rate
        
        for p in self.particles:
            # Calculate leveraged return
            leveraged_return = p.leverage * returns
            
            # Calculate score based on state and return
            if state == 0:  # Low volatility - focus on absolute returns
                score = leveraged_return
            elif state == 1:  # Medium volatility - balance return and risk
                score = leveraged_return - 0.5 * (leveraged_return ** 2)
            else:  # High volatility - more weight on risk
                score = leveraged_return - (leveraged_return ** 2)
                
            # Update weight
            p.weight *= np.exp(score)
            
        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
                
        # Resample and add noise
        self._resample()
        self._add_noise()
        
    def get_leverage_estimate(self) -> float:
        """Get current optimal leverage estimate"""
        weights = np.array([p.weight for p in self.particles])
        leverages = np.array([p.leverage for p in self.particles])
        return np.average(leverages, weights=weights)