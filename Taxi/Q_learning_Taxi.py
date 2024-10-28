import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

class TaxiQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.min_epsilon = min_epsilon  # Minimum exploration rate
        
        # Initialize Q-table
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # Action mappings for better visualization
        self.action_mapping = {
            0: "South",
            1: "North",
            2: "East",
            3: "West",
            4: "Pickup",
            5: "Dropoff"
        }
    
    def epsilon_greedy(self, state):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def decay_epsilon(self):
        """Decay epsilon with a minimum value."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def q_update(self, state, action, reward, next_state):
        """Perform Q-learning update."""
        # Q-learning uses max Q-value of next state (off-policy)
        next_max_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state, action]
        
        # Q-learning update rule
        self.q_table[state, action] = current_q + self.alpha * (
            reward + self.gamma * next_max_q - current_q
        )
    
    def train(self, n_episodes=10000, max_steps=200):
        """Train the agent using Q-learning algorithm."""
        # Initialize metrics
        episode_rewards = []
        episode_lengths = []
        best_reward = float('-inf')
        best_q_table = None
        success_rate_window = []
        avg_q_values = []
        
        for episode in tqdm(range(n_episodes), desc="Training Episodes"):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Select action using epsilon-greedy
                action = self.epsilon_greedy(state)
                
                # Take action and observe next state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update Q-table (key difference from SARSA)
                self.q_update(state, action, reward, next_state)
                
                total_reward += reward
                steps += 1
                
                if done:
                    success_rate_window.append(1 if reward == 20 else 0)
                    break
                    
                state = next_state
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Store episode statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Calculate average Q-values
            avg_q_values.append(np.mean(self.q_table))
            
            # Track best performance using moving average
            window_size = 100
            if episode >= window_size:
                moving_avg = np.mean(episode_rewards[-window_size:])
                if moving_avg > best_reward:
                    best_reward = moving_avg
                    best_q_table = np.copy(self.q_table)
            
            # Print progress every 1000 episodes
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_steps = np.mean(episode_lengths[-100:])
                success_rate = np.mean(success_rate_window[-100:]) if success_rate_window else 0
                print(f"\nEpisode {episode + 1}")
                print(f"Average Reward (last 100): {avg_reward:.2f}")
                print(f"Average Steps (last 100): {avg_steps:.2f}")
                print(f"Success Rate (last 100): {success_rate:.2%}")
                print(f"Current Epsilon: {self.epsilon:.3f}")
                print(f"Average Q-value: {avg_q_values[-1]:.3f}")
        
        # Use the best performing Q-table
        if best_q_table is not None:
            self.q_table = best_q_table
            
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_rate_window': success_rate_window,
            'avg_q_values': avg_q_values
        }
    
    def run_episode(self, render=True):
        """Run a single episode using the learned policy."""
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        actions_taken = []
        trajectory = []
        
        if render:
            print("\nStarting new episode...")
        
        while not done:
            # Select best action according to Q-table
            action = np.argmax(self.q_table[state])
            actions_taken.append(action)
            trajectory.append(state)
            
            if render:
                clear_output(wait=True)
                self.env.render()
                print(f"\nAction taken: {self.action_mapping[action]}")
            
            # Take action
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        trajectory.append(state)
        return total_reward, actions_taken, trajectory

def plot_training_results(results, window=100):
    """Plot comprehensive training results."""
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(results['episode_rewards'], alpha=0.5, label='Raw Rewards')
    plt.plot(np.convolve(results['episode_rewards'], 
                        np.ones(window)/window, 
                        mode='valid'), 
             label=f'Moving Average ({window} episodes)')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(results['episode_lengths'], alpha=0.5, label='Raw Lengths')
    plt.plot(np.convolve(results['episode_lengths'], 
                        np.ones(window)/window, 
                        mode='valid'), 
             label=f'Moving Average ({window} episodes)')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    # Plot success rate
    plt.subplot(2, 2, 3)
    success_rate = np.convolve(results['success_rate_window'], 
                              np.ones(window)/window, 
                              mode='valid')
    plt.plot(success_rate)
    plt.title(f'Success Rate (Moving Average {window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    # Plot average Q-values
    plt.subplot(2, 2, 4)
    plt.plot(results['avg_q_values'])
    plt.title('Average Q-values')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-value')
    
    plt.tight_layout()
    plt.show()

def evaluate_agent(agent, n_episodes=100, render=False):
    """Evaluate the agent's performance over multiple episodes."""
    evaluation_rewards = []
    evaluation_lengths = []
    successful_episodes = 0
    
    for _ in range(n_episodes):
        reward, actions, trajectory = agent.run_episode(render=render)
        evaluation_rewards.append(reward)
        evaluation_lengths.append(len(actions))
        if reward >= 20:  # Successful delivery
            successful_episodes += 1
    
    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"Average Reward: {np.mean(evaluation_rewards):.2f} ± {np.std(evaluation_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(evaluation_lengths):.2f} ± {np.std(evaluation_lengths):.2f}")
    print(f"Success Rate: {successful_episodes/n_episodes:.2%}")
    
    # Print policy analysis
    analyze_policy(agent)

def analyze_policy(agent):
    """Analyze the learned policy."""
    print("\nPolicy Analysis:")
    # Count how often each action is optimal in different states
    optimal_actions = np.argmax(agent.q_table, axis=1)
    action_counts = {action: np.sum(optimal_actions == action) 
                    for action in range(agent.n_actions)}
    
    total_states = len(optimal_actions)
    print("\nOptimal Action Distribution:")
    for action, count in action_counts.items():
        percentage = (count / total_states) * 100
        print(f"{agent.action_mapping[action]}: {percentage:.1f}% ({count} states)")

# Example usage
if __name__ == "__main__":
    # Create environment
    env = gym.make('Taxi-v3')
    
    # Create and train agent
    agent = TaxiQLearningAgent(env)
    training_results = agent.train(n_episodes=10000)
    
    # Plot results
    plot_training_results(training_results)
    
    # Evaluate agent
    evaluate_agent(agent)
    
    # Watch a single episode
    reward, actions, trajectory = agent.run_episode(render=True)
    print(f"\nEpisode finished with reward: {reward}")
    print(f"Actions taken: {[agent.action_mapping[a] for a in actions]}")
    
    # Close environment
    env.close()