from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class TrainingConfig:
    """
    Configuration for the Taxi-v3 Experiment.
    """
    # --- Environment Constants ---
    # We use Taxi-v3 as the chosen environment (Etapa 2)
    env_name: str = "Taxi-v3"
    env_kwargs: Dict = field(default_factory=dict)
    
    # --- Training Variables ---
    # Taxi is a navigation task; 25,000 episodes is usually sufficient for convergence.
    n_episodes: int = 25000          
    learning_rate: float = 0.1       # Default LR (will be overridden by main.py experiments)
    discount_factor: float = 0.99    # High discount because the goal (dropoff) is far in the future
    
    # --- Exploration Parameters ---
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    
    # Auto-calculate decay to finish exploring at 60% of training
    epsilon_decay: Optional[float] = None 
    
    # --- Visualization ---
    rolling_window: int = 100        # Smaller window than Blackjack because Taxi is less noisy
    plot_save_dir: str = "./plots_taxi"

    def __post_init__(self):
        """
        Auto-calculates decay based on the number of episodes.
        We explore for the first 60% of the training.
        """
        if self.epsilon_decay is None:
            decay_duration = self.n_episodes * 0.6
            self.epsilon_decay = (self.initial_epsilon - self.final_epsilon) / decay_duration