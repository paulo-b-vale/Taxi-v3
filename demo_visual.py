"""
Taxi-v3 Q-Learning Visual Demo
==============================
Watch the trained AI agent solve the Taxi environment!
The taxi will pick up passengers and deliver them to destinations.

Usage: python demo_visual.py
"""

import gymnasium as gym
import numpy as np
import time
from collections import defaultdict


class QLearningAgent:
    """Q-Learning agent for Taxi-v3"""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99,
                 initial_epsilon=1.0, epsilon_decay=0.0005, final_epsilon=0.01):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def get_action(self, obs, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs]))
    
    def update(self, obs, action, reward, terminated, next_obs):
        """Update Q-values using Bellman equation"""
        future_q = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q
        self.q_values[obs][action] += self.lr * (target - self.q_values[obs][action])
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def train_agent(n_episodes=5000, verbose=True):
    """Train Q-Learning agent"""
    env = gym.make('Taxi-v3')
    agent = QLearningAgent(env)
    
    rewards_history = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
            total_reward += reward
        
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # Print progress
        if verbose and (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    print(f"\n✓ Training complete! Final avg reward: {np.mean(rewards_history[-100:]):.2f}")
    return agent


def run_visual_demo(agent, n_games=10, delay=0.5):
    """Run visual demo of trained agent"""
    print(f"\n🚕 Starting visual demo ({n_games} games)...")
    print("Watch the taxi (yellow) pick up passengers and deliver them!\n")
    
    # Create environment with rendering
    env = gym.make('Taxi-v3', render_mode='human')
    
    for game in range(n_games):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"--- Game {game + 1}/{n_games} ---")
        
        while not done:
            action = agent.get_action(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            time.sleep(delay)  # Slow down for visualization
        
        print(f"Completed in {steps} steps | Reward: {total_reward}")
        time.sleep(1)  # Pause between games
    
    env.close()
    print("\n✓ Demo complete!")


def main():
    print("=" * 50)
    print("🚕 Taxi-v3 Q-Learning Visual Demo")
    print("=" * 50)
    
    # Train agent
    print("\n📚 Phase 1: Training the agent...")
    agent = train_agent(n_episodes=5000)
    
    # Run visual demo
    print("\n🎮 Phase 2: Visual demonstration...")
    run_visual_demo(agent, n_games=5, delay=0.3)


if __name__ == "__main__":
    main()
