import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from matplotlib.figure import Figure

class Visualizer:
    """
    Visualization tool for RL analysis (Etapa 4: Visualização dos Resultados).
    """
    
    def __init__(self, rolling_window: int = 500):
        self.rolling_window = rolling_window

    def get_moving_avgs(self, arr: List, window: int):
        """Smoothes noisy data using a moving average window."""
        if len(arr) == 0:
            return np.array([])
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode="valid"
        ) / window

    def compare_metrics(self, metrics_list: List[Dict], labels: List[str],
                       metric_name: str, title: str = "", 
                       save_path: Optional[str] = None) -> Figure:
        """
        Generates the critical comparison plot for the report.
        Overlays multiple experiments on one chart.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Crucial for Blackjack: Show the "Break Even" line (0.0)
        # If the agent is above this line, it is winning (impossible in long run)
        # If it is near this line (-0.05), it is playing optimally.
        if "reward" in metric_name:
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label="Break Even (0.0)")

        # Plot each experiment
        for metrics, label in zip(metrics_list, labels):
            if metric_name in metrics:
                data = metrics[metric_name]
                
                # Only plot if we have enough data for smoothing
                if len(data) >= self.rolling_window:
                    y = self.get_moving_avgs(data, self.rolling_window)
                    x = range(len(y))
                    
                    # Plot the line
                    ax.plot(x, y, label=label, linewidth=2, alpha=0.8)
                else:
                    print(f"Warning: Not enough data points in {label} for smoothing.")

        ax.set_title(title)
        ax.set_xlabel(f"Episode (Smoothed window: {self.rolling_window})")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to: {save_path}")
        
        return fig

    def print_training_summary(self, metrics: Dict[str, List]):
        """Simple console output for quick checks."""
        rewards = metrics['episode_rewards']
        if not rewards:
            return
            
        # Analyze last 10% of training to see final performance
        last_chunk = int(len(rewards) * 0.1)
        recent_rewards = rewards[-last_chunk:]
        
        avg_reward = np.mean(recent_rewards)
        win_rate = np.sum(np.array(recent_rewards) > 0) / len(recent_rewards)
        
        print(f"  Final Average Reward: {avg_reward:.4f}")
        print(f"  Final Win Rate: {win_rate:.1%}")