import gymnasium as gym
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
import numpy as np

class TrainingLoop:
    """
    Manages the Training Lifecycle (Etapa 3: Processo de Treinamento).
    Responsible for:
    1. Running Episodes
    2. Collecting Metrics
    3. Managing the Agent-Environment Loop
    """
    
    def __init__(self, agent, env: gym.Env, n_episodes: int, progress_bar: bool = True):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.progress_bar = progress_bar

    def run_training(self) -> Dict[str, List[Any]]:
        """
        Executes the main training loop.
        Returns a dictionary of metrics for Phase 4 (Analysis).
        """
        # Metrics Storage
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_errors': [] # Stores the MEAN error per episode
        }

        # Setup Progress Bar
        iterator = range(self.n_episodes)
        if self.progress_bar:
            iterator = tqdm(iterator, desc="Training")

        for _ in iterator:
            # --- Start of Episode ---
            obs, _ = self.env.reset()
            done = False
            
            total_reward = 0
            steps = 0
            episode_errors = [] # Temp list for just this episode

            while not done:
                # 1. Decide
                action = self.agent.get_action(obs)

                # 2. Act
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                # 3. Learn (and capture the error)
                td_error = self.agent.update(obs, action, reward, terminated, next_obs)
                episode_errors.append(np.abs(td_error)) # Use absolute error for magnitude

                # 4. Transition
                obs = next_obs
                done = terminated or truncated
                total_reward += reward
                steps += 1

            # --- End of Episode ---
            self.agent.decay_epsilon()

            # Log efficient metrics
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_lengths'].append(steps)
            
            # Calculate mean error for this episode (prevents memory leak)
            avg_error = np.mean(episode_errors) if episode_errors else 0
            metrics['episode_errors'].append(avg_error)

        return metrics