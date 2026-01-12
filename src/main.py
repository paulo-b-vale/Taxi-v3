#!/usr/bin/env python3
import os
import gymnasium as gym
from typing import Dict, Any
import copy

# Import our modular components
from agent import QLearningAgent
from trainer import TrainingLoop
from visualizer import Visualizer
from config import TrainingConfig

def run_single_experiment(config: TrainingConfig, label: str) -> Dict[str, Any]:
    """
    Executes one complete training run for Taxi.
    """
    print(f"\n--- Starting Experiment: {label} ---")
    
    # Load the Generic Environment (Taxi-v3)
    # We use standard gym.make instead of the specific Blackjack wrapper
    env = gym.make(config.env_name, **config.env_kwargs)
    
    # Add RecordEpisodeStatistics to ensure standard metric tracking
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Initialize Agent
    agent = QLearningAgent(
        env=env,
        learning_rate=config.learning_rate,
        initial_epsilon=config.initial_epsilon,
        epsilon_decay=config.epsilon_decay,
        final_epsilon=config.final_epsilon,
        discount_factor=config.discount_factor
    )
    
    # Initialize Trainer
    trainer = TrainingLoop(
        agent=agent,
        env=env,
        n_episodes=config.n_episodes,
        progress_bar=True
    )
    
    # Run Training
    metrics = trainer.run_training()
    env.close()
    
    return metrics

def main():
    # Base Config (defaults to Taxi settings from config.py)
    base_config = TrainingConfig()
    
    # Prepare output directory
    os.makedirs(base_config.plot_save_dir, exist_ok=True)
    viz = Visualizer(rolling_window=base_config.rolling_window)

    # --- Define The Experiments (Etapa 3/4) ---
    # We compare 3 Learning Rates to satisfy the assignment requirements.
    # For Taxi, very low LR (0.01) is usually too slow, and high LR (0.9) can be unstable.
    experiments = [
        {"lr": 0.01, "label": "Slow (LR=0.01)"},
        {"lr": 0.1,  "label": "Medium (LR=0.1)"}, 
        {"lr": 0.9,  "label": "Fast (LR=0.9)"}
    ]
    
    all_metrics = []
    labels = []

    # --- Execute Experiments ---
    for exp in experiments:
        # Create a fresh config copy for this specific run
        current_config = copy.copy(base_config)
        current_config.learning_rate = exp["lr"]
        
        # Recalculate epsilon decay based on the current config
        decay_duration = current_config.n_episodes * 0.6
        current_config.epsilon_decay = (current_config.initial_epsilon - current_config.final_epsilon) / decay_duration

        # Run the experiment
        metrics = run_single_experiment(current_config, exp["label"])
        all_metrics.append(metrics)
        labels.append(exp["label"])
        
        # Print a quick summary to the console
        viz.print_training_summary(metrics)

    # --- Generate Report Plots (Etapa 4) ---
    print("\nGenerating Taxi Analysis Plots...")
    
    # Plot 1: Rewards (Convergence check)
    viz.compare_metrics(
        all_metrics, labels, "episode_rewards",
        title=f"Taxi-v3: Learning Rate Comparison (Rewards)",
        save_path=os.path.join(base_config.plot_save_dir, "taxi_rewards.png")
    )
    
    # Plot 2: Episode Lengths (Efficiency check)
    # In Taxi, we want to see this line go DOWN (fewer steps to destination)
    viz.compare_metrics(
        all_metrics, labels, "episode_lengths",
        title="Taxi-v3: Steps to Reach Destination (Lower is Better)",
        save_path=os.path.join(base_config.plot_save_dir, "taxi_lengths.png")
    )
    
    print(f"DONE! Check the '{base_config.plot_save_dir}' folder for your report graphs.")

if __name__ == "__main__":
    main()