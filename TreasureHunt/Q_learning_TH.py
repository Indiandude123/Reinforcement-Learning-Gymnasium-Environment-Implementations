import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import TreasureHunt
import torch
# import numpy as np
from typing import Union, Optional

class QLearningAgent:
    def __init__(self, env, alpha=0.3, gamma=0.95, epsilon=0.4, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = self.initialize_q_table()
        
    def initialize_q_table(self):
        """Initialize Q-table with small random values for exploration."""
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        return np.random.uniform(low=0, high=0.1, size=(n_states, n_actions))
    
    def epsilon_greedy(self, state):
        """Epsilon-greedy action selection with decaying epsilon."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def decay_epsilon(self):
        """Decay epsilon with a minimum value."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def q_update(self, state, action, reward, next_state):
        """Perform Q-learning update."""
        # Q-learning uses max Q-value of next state instead of actual next action
        best_next_action = np.argmax(self.q_table[next_state])
        next_q = self.q_table[next_state, best_next_action]
        
        current_q = self.q_table[state, action]
        self.q_table[state, action] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )
        
    def train(self, n_episodes=1000, max_steps=500):
        """Train the agent using Q-learning."""
        episode_rewards = []
        episode_lengths = []
        best_episode_reward = float('-inf')
        best_q_table = None

        for episode in tqdm(range(n_episodes), desc="Training Episodes"):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Select action using epsilon-greedy
                action = self.epsilon_greedy(state)
                
                # Take action and observe next state and reward
                next_state, reward = self.env.step(action)
                
                # Update Q-table
                self.q_update(state, action, reward, next_state)
                
                episode_reward += reward
                steps += 1
                
                # Check if terminal state reached
                if self.check_terminal(next_state):
                    break
                    
                state = next_state
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Store episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # Keep track of best performance
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_q_table = np.copy(self.q_table)
        
        # Use the best performing Q-table
        self.q_table = best_q_table
        return self.q_table, episode_rewards, episode_lengths
    
    def check_terminal(self, state):
        """Check if the current state is terminal."""
        ship_location, treasure_locations = self.env.locations_from_state(state)
        
        # Terminal conditions
        if ship_location == self.env.locations['fort'][0]:
            return True
        if ship_location in self.env.locations['pirate']:
            return True
        if not treasure_locations and ship_location == self.env.locations['fort'][0]:
            return True
            
        return False

    def run_episode(self, render=True):
        """Run a single episode using the learned Q-table."""
        state = self.env.reset()
        frames = [self.env.render()] if render else []
        total_reward = 0
        actions_taken = []
        
        while True:
            # Select best action according to Q-table
            action = np.argmax(self.q_table[state])
            actions_taken.append(action)
            
            # Take action
            next_state, reward = self.env.step(action)
            
            if render:
                frames.append(self.env.render())
            
            total_reward += reward
            
            if self.check_terminal(next_state):
                break
                
            state = next_state
            
        return frames, total_reward, actions_taken

def plot_training_results(episode_rewards, episode_lengths, window=50):
    """Plot smoothed learning curves with confidence intervals."""
    plt.figure(figsize=(15, 5))
    
    # Calculate rolling statistics
    def rolling_stats(data, window):
        rolling_mean = np.convolve(data, np.ones(window)/window, mode='valid')
        return rolling_mean
    
    # Plot rewards
    plt.subplot(1, 3, 1)
    rewards_smooth = rolling_stats(episode_rewards, window)
    episodes = range(len(rewards_smooth))
    plt.plot(episodes, rewards_smooth, label='Average Reward')
    plt.plot(episode_rewards, alpha=0.3, label='Raw Rewards')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot episode lengths
    plt.subplot(1, 3, 2)
    lengths_smooth = rolling_stats(episode_lengths, window)
    plt.plot(episodes, lengths_smooth, label='Average Length')
    plt.plot(episode_lengths, alpha=0.3, label='Raw Lengths')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    # Plot reward distribution
    plt.subplot(1, 3, 3)
    plt.hist(episode_rewards, bins=50, alpha=0.7)
    plt.axvline(np.mean(episode_rewards), color='r', linestyle='dashed', 
                label=f'Mean: {np.mean(episode_rewards):.2f}')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_agent(agent, n_episodes=100):
    """Evaluate the agent's performance over multiple episodes."""
    evaluation_rewards = []
    evaluation_lengths = []
    
    for _ in range(n_episodes):
        _, reward, actions = agent.run_episode(render=False)
        evaluation_rewards.append(reward)
        evaluation_lengths.append(len(actions))
    
    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"Average Reward: {np.mean(evaluation_rewards):.2f} ± {np.std(evaluation_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(evaluation_lengths):.2f} ± {np.std(evaluation_lengths):.2f}")


def save_q_table(q_table: Union[np.ndarray, torch.Tensor], 
                filepath: str,
                env_name: Optional[str] = None) -> None:
    """
    Save Q-table as a PyTorch .pt file
    
    Args:
        q_table: Q-table as either numpy array or PyTorch tensor
        filepath: Path where to save the Q-table
        env_name: Optional environment name to include in metadata
    """
    # Convert to PyTorch tensor if needed
    if isinstance(q_table, np.ndarray):
        q_table = torch.from_numpy(q_table).float()
    elif not isinstance(q_table, torch.Tensor):
        raise TypeError("Q-table must be either numpy array or PyTorch tensor")
    
    # Ensure the file has .pt extension
    if not filepath.endswith('.pt'):
        filepath += '.pt'
    
    # Save the Q-table
    torch.save(q_table, filepath)
    print(f"Q-table saved successfully to {filepath}")
    
    # Print Q-table information
    print(f"\nQ-table information:")
    print(f"Shape: {q_table.shape}")
    print(f"Data type: {q_table.dtype}")
    print(f"Device: {q_table.device}")

def load_q_table(filepath: str) -> torch.Tensor:
    """
    Load Q-table from a .pt file
    
    Args:
        filepath: Path to the saved Q-table
        
    Returns:
        The loaded Q-table as a PyTorch tensor
    """
    # Ensure the file has .pt extension
    if not filepath.endswith('.pt'):
        filepath += '.pt'
    
    # Load the Q-table
    q_table = torch.load(filepath)
    print(f"Q-table loaded successfully from {filepath}")
    
    # Print Q-table information
    print(f"\nQ-table information:")
    print(f"Shape: {q_table.shape}")
    print(f"Data type: {q_table.dtype}")
    print(f"Device: {q_table.device}")
    
    return q_table

# Example usage for different environments
def save_environment_q_tables(agent, env_name: str) -> None:
    """
    Save Q-tables for a specific environment
    
    Args:
        agent: The RL agent containing the Q-table
        env_name: Name of the environment
    """
    # Get Q-table from agent
    q_table = agent.q_table
    
    # Generate appropriate filename
    filename = f"q_table_{env_name.lower()}.pt"
    
    # Save Q-table
    save_q_table(q_table, filename, env_name)

# Example usage
if __name__ == "__main__":
    # Define environment
    locations = {
        'ship': [(0,0)],
        'land': [(3,0), (3,1), (3,2), (4,2), (4,1), (5,2), 
                (0,7), (0,8), (0,9), (1,7), (1,8), (2,7)],
        'fort': [(9,9)],
        'pirate': [(4,7), (8,5)],
        'treasure': [(4,0), (1,9)]
    }
    
    # Create and train agent
    env = TreasureHunt(locations)
    agent = QLearningAgent(env)
    
    # Train the agent
    q_table, rewards, lengths = agent.train(n_episodes=10000)
    
    save_q_table(agent.q_table, 'treasureHunt_q_table.pt', 'TreasureHunt_v1')
    # Plot training results
    plot_training_results(rewards, lengths)
    
    # Evaluate the trained agent
    evaluate_agent(agent)
     
    # # Visualize a single episode
    # frames, reward, actions = agent.run_episode()
    # print(f"\nFinal episode reward: {reward}")
    # print(f"Actions taken: {actions}")
    
    