# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from env import TreasureHunt
# import os
# import cv2

# class SARSAAgent:
#     def __init__(self, env, alpha=0.3, gamma=0.95, epsilon=0.4, epsilon_decay=0.95):
#         self.env = env
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.q_table = self.initialize_q_table()
        
#     def initialize_q_table(self):
#         """Initialize Q-table with small random values for exploration."""
#         n_states = self.env.observation_space.n
#         n_actions = self.env.action_space.n
#         return np.random.uniform(low=0, high=0.1, size=(n_states, n_actions))
    
#     def epsilon_greedy(self, state):
#         """Epsilon-greedy action selection with decaying epsilon."""
#         if np.random.random() < self.epsilon:
#             return self.env.action_space.sample()
#         else:
#             return np.argmax(self.q_table[state])
        
#     def sarsa_update(self, state, action, reward, next_state, next_action):
#         """Perform SARSA update on Q-table."""
#         current_q = self.q_table[state, action]
#         next_q = self.q_table[next_state, next_action]
#         self.q_table[state, action] = current_q + self.alpha * (
#             reward + self.gamma * next_q - current_q
#         )
        
#     def train(self, n_episodes=1000, max_steps=1000):
#         """Train the agent using the SARSA algorithm."""
#         episode_rewards = []
#         episode_lengths = []

#         for episode in tqdm(range(n_episodes), desc="Training Episodes"):
#             state = self.env.reset()
#             action = self.epsilon_greedy(state)
#             episode_reward = 0
#             steps = 0
            
#             for step in range(max_steps):
#                 next_state, reward = self.env.step(action)
#                 next_action = self.epsilon_greedy(next_state)
                
#                 self.sarsa_update(state, action, reward, next_state, next_action)
                
#                 episode_reward += reward
#                 steps += 1
                
#                 # Check terminal conditions
#                 if self.check_terminal(next_state):
#                     break
                    
#                 state, action = next_state, next_action
            
#             # Decay epsilon
#             self.epsilon *= self.epsilon_decay
            
#             episode_rewards.append(episode_reward)
#             episode_lengths.append(steps)
            
#         return self.q_table, episode_rewards, episode_lengths
    
#     def check_terminal(self, state):
#         """Check if the current state is terminal."""
#         ship_location, treasure_locations = self.env.locations_from_state(state)
        
#         # Terminal conditions:
#         # 1. Ship reaches fort
#         if ship_location == self.env.locations['fort'][0]:
#             return True
            
#         # 2. Ship hits a pirate
#         if ship_location in self.env.locations['pirate']:
#             return True
            
#         # 3. No more treasures and ship at fort
#         if not treasure_locations and ship_location == self.env.locations['fort'][0]:
#             return True
            
#         return False
    
#     def run_episode(self, render=True, save_gif=False, gif_path="episode.gif"):
#         """Run a single episode using the learned Q-table and optionally save as GIF."""
#         state = self.env.reset()
#         frames = []
#         total_reward = 0
#         actions_taken = []

#         while True:
#             # Select best action according to Q-table
#             action = np.argmax(self.q_table[state])
#             actions_taken.append(action)

#             # Take action
#             next_state, reward = self.env.step(action)

#             # Render and store frames if render=True
#             if render:
#                 frame = self.env.render()  # Assuming this returns an image array (RGB)
#                 if isinstance(frame, np.ndarray):
#                     frames.append(frame)
#                 else:
#                     print("Error: Rendered frame is not a valid numpy array.")
#                     return

#             total_reward += reward

#             if self.check_terminal(next_state):
#                 break

#             state = next_state

#         # Save the GIF if required
#         if save_gif:
#             self.save_frames_as_gif(frames, gif_path)

#         return frames, total_reward, actions_taken

#     def save_frames_as_gif(self, frames, gif_path):
#         """Save the frames as a GIF using OpenCV."""
#         temp_images = []

#         # Ensure all frames are in correct format
#         for i, frame in enumerate(frames):
#             # Convert RGB to BGR for OpenCV
#             frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             temp_images.append(frame_bgr)

#         # Save as video file if necessary
#         height, width, _ = temp_images[0].shape
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Set codec for GIF
#         video = cv2.VideoWriter('temp.avi', fourcc, 1, (width, height))

#         # Write each frame to the video
#         for frame in temp_images:
#             video.write(frame)

#         # Release the video writer
#         video.release()

#         # Convert the video to GIF using OpenCV and external tool
#         self.convert_video_to_gif('temp.avi', gif_path)

#     def convert_video_to_gif(self, video_path, gif_path):
#         """Use OpenCV to convert the video to a GIF."""
#         # Convert video to GIF using external conversion tool
#         try:
#             os.system(f'ffmpeg -i {video_path} {gif_path}')  # Using ffmpeg to convert
#             print(f"GIF saved successfully at {gif_path}")
#         except Exception as e:
#             print(f"Error converting video to GIF: {e}")


# def plot_learning_curve(episode_rewards, episode_lengths, window=50):
#     """Plot smoothed learning curves."""
#     plt.figure(figsize=(12, 5))
    
#     # Smooth the curves using moving average
#     smooth_rewards = np.convolve(episode_rewards, 
#                                 np.ones(window)/window, 
#                                 mode='valid')
#     smooth_lengths = np.convolve(episode_lengths, 
#                                 np.ones(window)/window, 
#                                 mode='valid')
    
#     plt.subplot(1, 2, 1)
#     plt.plot(smooth_rewards, label='Smoothed')
#     plt.plot(episode_rewards, alpha=0.3, label='Raw')
#     plt.title("Episode Rewards")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(smooth_lengths, label='Smoothed')
#     plt.plot(episode_lengths, alpha=0.3, label='Raw')
#     plt.title("Episode Lengths")
#     plt.xlabel("Episode")
#     plt.ylabel("Number of Steps")
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()

# # Example usage:
# if __name__ == "__main__":
#     locations = {
#         'ship': [(0,0)],
#         'land': [(3,0), (3,1), (3,2), (4,2), (4,1), (5,2), 
#                 (0,7), (0,8), (0,9), (1,7), (1,8), (2,7)],
#         'fort': [(9,9)],
#         'pirate': [(4,7), (8,5)],
#         'treasure': [(4,0), (1,9)]
#     }
    
#     env = TreasureHunt(locations)
#     agent = SARSAAgent(env)
#     # Train the agent
#     q_table, rewards, lengths = agent.train(n_episodes=13000)

#     # Visualize a single episode and save it as a GIF
#     frames, reward, actions = agent.run_episode(save_gif=True, gif_path="treasure_hunt_sarsa.gif")
#     print(f"\nFinal episode reward: {reward}")
#     print(f"Actions taken: {actions}")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import TreasureHunt
import os
import cv2
import pickle  # Import pickle for saving and loading the Q-table
import torch
# import numpy as np
from typing import Union, Optional

class SARSAAgent:
    def __init__(self, env, alpha=0.3, gamma=0.95, epsilon=0.4, epsilon_decay=0.95):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
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
        
    def sarsa_update(self, state, action, reward, next_state, next_action):
        """Perform SARSA update on Q-table."""
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        self.q_table[state, action] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )
        
    def train(self, n_episodes=1000, max_steps=1000, save_q_table=False, q_table_path="q_table.pkl"):
        """Train the agent using the SARSA algorithm."""
        episode_rewards = []
        episode_lengths = []

        for episode in tqdm(range(n_episodes), desc="Training Episodes"):
            state = self.env.reset()
            action = self.epsilon_greedy(state)
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                next_state, reward = self.env.step(action)
                next_action = self.epsilon_greedy(next_state)
                
                self.sarsa_update(state, action, reward, next_state, next_action)
                
                episode_reward += reward
                steps += 1
                
                # Check terminal conditions
                if self.check_terminal(next_state):
                    break
                    
                state, action = next_state, next_action
            
            # Decay epsilon
            self.epsilon *= self.epsilon_decay
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

        # Save the Q-table if required
        if save_q_table:
            self.save_q_table(q_table_path)
            
        return self.q_table, episode_rewards, episode_lengths
    
    def check_terminal(self, state):
        """Check if the current state is terminal."""
        ship_location, treasure_locations = self.env.locations_from_state(state)
        
        # Terminal conditions:
        # 1. Ship reaches fort
        if ship_location == self.env.locations['fort'][0]:
            return True
            
        # 2. Ship hits a pirate
        if ship_location in self.env.locations['pirate']:
            return True
            
        # 3. No more treasures and ship at fort
        if not treasure_locations and ship_location == self.env.locations['fort'][0]:
            return True
            
        return False

    def save_q_table(self, q_table_path):
        """Save the Q-table to a file using pickle."""
        with open(q_table_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved successfully to {q_table_path}")
    
    def load_q_table(self, q_table_path):
        """Load the Q-table from a file using pickle."""
        with open(q_table_path, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded successfully from {q_table_path}")
    
    def run_episode(self, render=True, save_gif=False, gif_path="episode.gif"):
        """Run a single episode using the learned Q-table and optionally save as GIF."""
        state = self.env.reset()
        frames = []
        total_reward = 0
        actions_taken = []

        while True:
            # Select best action according to Q-table
            action = np.argmax(self.q_table[state])
            actions_taken.append(action)

            # Take action
            next_state, reward = self.env.step(action)

            # Render and store frames if render=True
            if render:
                frame = self.env.render()  # Assuming this returns an image array (RGB)
                if isinstance(frame, np.ndarray):
                    frames.append(frame)
                else:
                    print("Error: Rendered frame is not a valid numpy array.")
                    return

            total_reward += reward

            if self.check_terminal(next_state):
                break

            state = next_state

        # Save the GIF if required
        if save_gif:
            self.save_frames_as_gif(frames, gif_path)

        return frames, total_reward, actions_taken

    def save_frames_as_gif(self, frames, gif_path):
        """Save the frames as a GIF using OpenCV."""
        temp_images = []

        # Ensure all frames are in correct format
        for i, frame in enumerate(frames):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            temp_images.append(frame_bgr)

        # Save as video file if necessary
        height, width, _ = temp_images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Set codec for GIF
        video = cv2.VideoWriter('temp.avi', fourcc, 1, (width, height))

        # Write each frame to the video
        for frame in temp_images:
            video.write(frame)

        # Release the video writer
        video.release()

        # Convert the video to GIF using OpenCV and external tool
        self.convert_video_to_gif('temp.avi', gif_path)

    def convert_video_to_gif(self, video_path, gif_path):
        """Use OpenCV to convert the video to a GIF."""
        try:
            os.system(f'ffmpeg -i {video_path} {gif_path}')  # Using ffmpeg to convert
            print(f"GIF saved successfully at {gif_path}")
        except Exception as e:
            print(f"Error converting video to GIF: {e}")


def plot_learning_curve(episode_rewards, episode_lengths, window=50):
    """Plot smoothed learning curves."""
    plt.figure(figsize=(12, 5))
    
    # Smooth the curves using moving average
    smooth_rewards = np.convolve(episode_rewards, 
                                np.ones(window)/window, 
                                mode='valid')
    smooth_lengths = np.convolve(episode_lengths, 
                                np.ones(window)/window, 
                                mode='valid')
    
    plt.subplot(1, 2, 1)
    plt.plot(smooth_rewards, label='Smoothed')
    plt.plot(episode_rewards, alpha=0.3, label='Raw')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(smooth_lengths, label='Smoothed')
    plt.plot(episode_lengths, alpha=0.3, label='Raw')
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


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

# Example usage:
if __name__ == "__main__":
    locations = {
        'ship': [(0, 0)],
        'land': [(3, 0), (3, 1), (3, 2), (4, 2), (4, 1), (5, 2), 
                (0, 7), (0, 8), (0, 9), (1, 7), (1, 8), (2, 7)],
        'fort': [(9, 9)],
        'pirate': [(4, 7), (8, 5)],
        'treasure': [(4, 0), (1, 9)]
    }
    
    env = TreasureHunt(locations)
    agent = SARSAAgent(env)

    # Train the agent and save the Q-table
    q_table, rewards, lengths = agent.train(n_episodes=13000, save_q_table=True, q_table_path="sarsa_q_table.pkl")

    save_q_table(agent.q_table, 'treasureHunt_sarsa_q_table.pt', 'TreasureHunt_v1')

    plot_learning_curve(rewards, lengths)
    
    # Visualize a single episode and save it as a GIF
    frames, reward, actions = agent.run_episode(save_gif=True, gif_path="treasure_hunt_sarsa.gif")
    print(f"\nFinal episode reward: {reward}")
    print(f"Actions taken: {actions}")

    # loaded_q_table = load_q_table('taxi_q_table.pt')
    # # To load the saved Q-table later:
    # agent.load_q_table("sarsa_q_table.pkl")
    # frames, reward, actions = agent.run_episode(save_gif=False)
    # print(f"\nFinal episode reward after loading Q-table: {reward}")
