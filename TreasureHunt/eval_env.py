import torch
import numpy as np
import gymnasium as gym
import moviepy.editor as mpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from env import TreasureHunt


class PolicyEvaluator:
    def __init__(self, env_name: str, q_table_path: str = None, q_table: Union[torch.Tensor, np.ndarray] = None):
        """
        Initialize the evaluator with either a path to a Q-table or a Q-table directly
        
        Args:
            env_name (str): Name of the gymnasium environment
            q_table_path (str, optional): Path to saved Q-table
            q_table (Union[torch.Tensor, np.ndarray], optional): Q-table directly
        """
        self.env = TreasureHunt
        
        # Load Q-table
        if q_table_path is not None:
            self.q_table = torch.load(q_table_path)
        elif q_table is not None:
            self.q_table = q_table if isinstance(q_table, torch.Tensor) else torch.from_numpy(q_table)
        else:
            raise ValueError("Either q_table_path or q_table must be provided")
            
    def evaluate_policy(self, n_episodes: int = 100, render: bool = False) -> Tuple[float, float]:
        """
        Evaluate the policy over multiple episodes
        
        Args:
            n_episodes (int): Number of episodes to evaluate
            render (bool): Whether to render the environment
            
        Returns:
            Tuple[float, float]: Average reward and average steps
        """
        total_rewards = []
        total_steps = []
        
        for _ in tqdm(range(n_episodes), desc="Evaluating Policy"):
            rewards, steps = self._run_single_episode(render=False)
            total_rewards.append(rewards)
            total_steps.append(steps)
            
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        std_reward = np.std(total_rewards)
        std_steps = np.std(total_steps)
        
        print(f"\nEvaluation Results over {n_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Steps: {avg_steps:.2f} ± {std_steps:.2f}")
        
        return avg_reward, avg_steps
    
    def visualize_trajectories(self, n_episodes: int = 5, max_steps: int = 500, 
                             fps: int = 1, output_prefix: str = "trajectory") -> None:
        """
        Visualize and save trajectories as videos
        
        Args:
            n_episodes (int): Number of episodes to visualize
            max_steps (int): Maximum steps per episode
            fps (int): Frames per second in output video
            output_prefix (str): Prefix for output video files
        """
        for i_episode in range(n_episodes):
            print(f"\nGenerating trajectory for episode {i_episode + 1}...")
            
            # Run episode and collect frames
            rewards, steps, frames = self._run_single_episode(render=True, 
                                                            collect_frames=True, 
                                                            max_steps=max_steps)
            
            # Save video
            output_path = f"{output_prefix}_episode_{i_episode + 1}.mp4"
            clip = mpy.ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(output_path, codec="libx264", verbose=False, logger=None)
            
            print(f"Episode {i_episode + 1} Results:")
            print(f"Total Reward: {rewards}")
            print(f"Total Steps: {steps}")
            print(f"Video saved to: {output_path}")
            
    def _run_single_episode(self, render: bool = False, collect_frames: bool = False,
                           max_steps: int = 500) -> Union[Tuple[float, int], Tuple[float, int, List]]:
        """
        Run a single episode
        
        Args:
            render (bool): Whether to render the environment
            collect_frames (bool): Whether to collect frames for visualization
            max_steps (int): Maximum steps before termination
            
        Returns:
            Union[Tuple[float, int], Tuple[float, int, List]]: 
                Returns rewards and steps, and optionally frames
        """
        state = self.env.reset()
        rewards, steps = 0, 0
        frames = [] if collect_frames else None
        
        while True:
            if collect_frames:
                frames.append(self.env.render())
                
            # Get action from Q-table
            action = torch.argmax(self.q_table[state]).item()
            
            # Take step in environment
            state, reward, terminated, truncated, _ = self.env.step(action)
            
            rewards += reward
            steps += 1
            
            # Check termination conditions
            if terminated or truncated or steps >= max_steps:
                if collect_frames:
                    frames.append(self.env.render())
                break
                
        if collect_frames:
            return rewards, steps, frames
        return rewards, steps

def plot_evaluation_results(rewards: List[float], steps: List[float]) -> None:
    """
    Plot evaluation results
    
    Args:
        rewards (List[float]): List of rewards per episode
        steps (List[float]): List of steps per episode
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot steps
    ax2.plot(steps)
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Steps')
    
    plt.tight_layout()
    plt.show()

# Example usage
def evaluate_environment(env_name: str, q_table_path: str, n_eval_episodes: int = 100, 
                        n_visual_episodes: int = 5) -> None:
    """
    Evaluate and visualize policy for a given environment
    
    Args:
        env_name (str): Name of the gymnasium environment
        q_table_path (str): Path to saved Q-table
        n_eval_episodes (int): Number of episodes for evaluation
        n_visual_episodes (int): Number of episodes to visualize
    """
    evaluator = PolicyEvaluator(env_name, q_table_path)
    
    # Evaluate policy
    print(f"\nEvaluating {env_name}...")
    avg_reward, avg_steps = evaluator.evaluate_policy(n_episodes=n_eval_episodes)
    
    # Visualize trajectories
    print(f"\nGenerating trajectory visualizations for {env_name}...")
    evaluator.visualize_trajectories(n_episodes=n_visual_episodes, 
                                   output_prefix=f"{env_name.lower()}_trajectory")
    
    
evaluate_environment(
    env_name='TreasureHunt-v1',
    q_table_path='treasureHunt_q_table.pt',  # Replace with your saved Q-table path
    n_eval_episodes=100,
    n_visual_episodes=5
)