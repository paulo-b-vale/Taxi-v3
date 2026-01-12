import gymnasium as gym
from typing import Tuple, Any

class EnvironmentWrapper:
    """
    Base wrapper for standardizing environment interactions.
    This satisfies 'Etapa 2: Escolha ou Construção do Ambiente'.
    """
    
    def __init__(self, env_name: str, **env_kwargs):
        self.env_name = env_name
        # Initialize Gymnasium environment
        # We adds RecordEpisodeStatistics to automatically track rewards/lengths
        self.env = gym.wrappers.RecordEpisodeStatistics(
            gym.make(env_name, **env_kwargs)
        )
        
    def get_env(self) -> gym.Env:
        return self.env
    
    def reset(self) -> Tuple[Any, dict]:
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()


class BlackjackEnvironmentWrapper(EnvironmentWrapper):
    """
    Specific implementation for Blackjack (Etapa 1: MDP Formulation).
    
    MDP Definitions for your Report:
    -------------------------------
    1. State Space (S): A tuple of 3 values (32 x 10 x 2 = 640 states)
       - Player Sum: 4 to 21 (Integers)
       - Dealer Card: 1 to 10 (Integers, 1 is Ace)
       - Usable Ace: 0 or 1 (Boolean)
       
    2. Action Space (A): Discrete(2)
       - 0: Stick (Stop taking cards)
       - 1: Hit (Request another card)
       
    3. Reward Function (R):
       - +1.0: Win
       - -1.0: Loss (or Bust)
       -  0.0: Draw
       
    4. Transitions (P):
       - Stochastic (Random) based on infinite deck drawing probabilities.
    """
    
    def __init__(self, sab: bool = False):
        # SAB = Sutton and Barto (Standard RL Textbook rules)
        super().__init__("Blackjack-v1", sab=sab)
        
    def get_state_description(self, state: Tuple[int, int, bool]) -> str:
        """Helper for debugging/visualization."""
        player_sum, dealer_card, usable_ace = state
        ace_str = "Usable Ace" if usable_ace else "No Ace"
        return f"Player:{player_sum} vs Dealer:{dealer_card} ({ace_str})"

    def get_action_name(self, action: int) -> str:
        return "Hit" if action == 1 else "Stick"