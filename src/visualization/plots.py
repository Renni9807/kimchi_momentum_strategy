# visualization/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

def plot_sharpe_heatmap(results_matrix: np.ndarray, 
                       X_list: Union[List[float], np.ndarray], 
                       Y_list: Union[List[float], np.ndarray], 
                       save_path: Union[str, Path],
                       title: Optional[str] = None) -> None:
    """
    Plot enhanced Sharpe ratio heatmap with best parameter identification
    
    Args:
        results_matrix: 2D numpy array of Sharpe ratios
        X_list: List of X threshold values
        Y_list: List of Y threshold values 
        save_path: Path to save the plot
        title: Optional title for the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Find best parameters
    best_idx = np.unravel_index(np.argmax(results_matrix), results_matrix.shape)
    best_sharpe = results_matrix[best_idx]
    best_X = X_list[best_idx[1]]
    best_Y = Y_list[best_idx[0]]
    
    # Create heatmap with improved visualization
    sns.heatmap(results_matrix, 
                xticklabels=[f'{x:.2%}' for x in X_list],
                yticklabels=[f'{y:.2%}' for y in Y_list],
                cmap='RdYlGn',  # Red-Yellow-Green colormap
                center=np.mean(results_matrix),  # Center colormap at mean
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Sharpe Ratio'})
    
    # Mark the best parameter combination
    plt.plot(best_idx[1] + 0.5, best_idx[0] + 0.5, 'r*', markersize=15,
             label=f'Best: X={best_X:.2%}, Y={best_Y:.2%}\nSharpe={best_sharpe:.3f}')
    
    # Add contour lines for better visualization of Sharpe ratio levels
    plt.contour(np.arange(len(X_list)) + 0.5,
                np.arange(len(Y_list)) + 0.5,
                results_matrix,
                colors='black',
                alpha=0.3,
                linestyles='dashed')
    
    if title:
        plt.title(title, fontsize=14, pad=20)
    else:
        plt.title('Parameter Search Results - Sharpe Ratio', fontsize=14, pad=20)
    
    plt.xlabel('X Threshold (Upward Movement)', fontsize=12)
    plt.ylabel('Y Threshold (Downward Movement)', fontsize=12)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    # Print top 5 combinations
    top_5_idx = np.argsort(results_matrix.ravel())[-5:][::-1]
    top_5_positions = [np.unravel_index(idx, results_matrix.shape) for idx in top_5_idx]
    
    textstr = 'Top 5 Combinations:\n'
    for i, (y_idx, x_idx) in enumerate(top_5_positions, 1):
        textstr += f'{i}. X={X_list[x_idx]:.2%}, Y={Y_list[y_idx]:.2%}, '
        textstr += f'Sharpe={results_matrix[y_idx, x_idx]:.3f}\n'
    
    plt.figtext(1.15, 0.5, textstr, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                pad_inches=1)  # Extra padding for annotations
    plt.close()

def plot_equity_curve(equity_curves: List[pd.Series], 
                     save_path: Union[str, Path],
                     title: Optional[str] = None,
                     show_percentiles: bool = True) -> None:
    """
    Plot equity curves with optional percentile bands
    
    Args:
        equity_curves: List of equity curve series
        save_path: Path to save the plot
        title: Optional title for the plot
        show_percentiles: Whether to show 25th and 75th percentile bands
    """
    plt.figure(figsize=(12, 6))
    
    # Convert all curves to numpy arrays for easier manipulation
    curve_arrays = [curve.values for curve in equity_curves]
    dates = equity_curves[0].index
    
    # Plot individual curves with low opacity
    for curve in curve_arrays:
        plt.plot(dates, curve, alpha=0.15, color='blue', linewidth=0.5)
    
    # Calculate and plot mean curve
    mean_curve = np.mean(curve_arrays, axis=0)
    plt.plot(dates, mean_curve, 'b-', linewidth=2, label='Mean Return')
    
    if show_percentiles:
        # Calculate and plot percentile bands
        percentile_25 = np.percentile(curve_arrays, 25, axis=0)
        percentile_75 = np.percentile(curve_arrays, 75, axis=0)
        
        plt.fill_between(dates, percentile_25, percentile_75, 
                        color='blue', alpha=0.15, 
                        label='25th-75th Percentile')
    
    if title:
        plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_returns_distribution(returns: pd.Series,
                            save_path: Union[str, Path],
                            title: Optional[str] = None) -> None:
    """
    Plot distribution of strategy returns with normal distribution overlay
    
    Args:
        returns: Series of strategy returns
        save_path: Path to save the plot
        title: Optional title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot returns distribution
    sns.histplot(returns, stat='density', bins=50, alpha=0.6)
    
    # Add normal distribution overlay
    x = np.linspace(returns.min(), returns.max(), 100)
    y = pd.Series(x).apply(lambda x: np.exp(-(x - returns.mean())**2 / 
                                           (2 * returns.std()**2)) / 
                                           (returns.std() * np.sqrt(2 * np.pi)))
    plt.plot(x, y, 'r--', label='Normal Distribution')
    
    if title:
        plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Daily Returns', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_drawdown_periods(equity_curve: pd.Series,
                         save_path: Union[str, Path],
                         title: Optional[str] = None) -> None:
    """
    Plot drawdown periods analysis
    
    Args:
        equity_curve: Series of portfolio values
        save_path: Path to save the plot
        title: Optional title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate drawdown series
    rolling_max = equity_curve.expanding(min_periods=1).max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    
    # Plot equity curve
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(equity_curve.index, equity_curve.values, 'b-', label='Portfolio Value')
    ax2.fill_between(drawdowns.index, drawdowns.values * 100, 0, 
                     color='red', alpha=0.3, label='Drawdown')
    
    if title:
        plt.title(title, fontsize=14, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value', fontsize=12, color='b')
    ax2.set_ylabel('Drawdown %', fontsize=12, color='r')
    
    # Adjust gridlines and legend
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()