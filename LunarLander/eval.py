import os
import cv2
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import yaml
import argparse
import zipfile
from collections import deque
from agent import Agent

def load_parameters(file_path):
    """
    Load parameters from a YAML configuration file.
    
    Args:
        file_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Dictionary containing the configuration parameters
    """
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def eval_agent(agent, env, num_episodes=5):
    """
    Evaluate the trained agent's performance and record videos/frames of the episodes.
    
    Args:
        agent (Agent): The trained agent to evaluate
        env (gym.Env): The environment to evaluate in
        num_episodes (int): Number of evaluation episodes to run
        
    Returns:
        list: List of scores from evaluation episodes
    """
    scores = []
    video_dir = os.path.join(os.getcwd(), 'videos')
    os.makedirs(video_dir, exist_ok=True)
    
    # Test available codecs
    test_codecs = [
        ('mp4v', '.mp4'),  # MPEG-4
        ('XVID', '.avi'),  # XVID
        ('MJPG', '.avi'),  # Motion JPEG
    ]
    
    # Get initial frame dimensions
    state = env.reset()[0]
    dummy_frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    height, width = dummy_frame.shape[:2]
    
    # Find first working codec
    working_codec = None
    working_ext = None
    
    for codec, ext in test_codecs:
        test_path = os.path.join(video_dir, f'test{ext}')
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(
                test_path,
                fourcc,
                30.0,
                (width, height)
            )
            if test_writer.isOpened():
                working_codec = codec
                working_ext = ext
                test_writer.release()
                os.remove(test_path)
                print(f"Using codec: {codec} with extension {ext}")
                break
            test_writer.release()
        except Exception:
            continue
    
    if working_codec is None:
        print("Warning: No working video codec found. Will save individual frames instead.")
    
    # Run evaluation episodes
    for i_episode in range(num_episodes):
        state = env.reset()[0]
        score = 0
        frames = []
        
        # Run single episode
        while True:
            action = agent.act(state, eps=0.0)
            frame = env.render()
            
            if frame is not None:
                frame = (frame * 255).astype('uint8') if frame.dtype == 'float32' else frame
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)
                
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            
            if terminated or truncated:
                break
                
        scores.append(score)
        
        # Save episode recordings
        try:
            if len(frames) > 0:
                print(f"Episode {i_episode + 1}: {len(frames)} frames captured")
                
                if working_codec:
                    # Save as video
                    video_path = os.path.join(video_dir, f'eval_episode_{i_episode}{working_ext}')
                    fourcc = cv2.VideoWriter_fourcc(*working_codec)
                    out = cv2.VideoWriter(
                        video_path,
                        fourcc,
                        30.0,
                        (width, height)
                    )
                    
                    for frame in frames:
                        out.write(frame)
                    out.release()
                else:
                    # Save individual frames
                    episode_dir = os.path.join(video_dir, f'episode_{i_episode}')
                    os.makedirs(episode_dir, exist_ok=True)
                    for frame_idx, frame in enumerate(frames):
                        frame_path = os.path.join(episode_dir, f'frame_{frame_idx:04d}.png')
                        cv2.imwrite(frame_path, frame)
                
                print(f'Episode {i_episode + 1} Score: {score}')
            else:
                print(f"No frames captured for episode {i_episode}")
        except Exception as e:
            print(f"Error in episode {i_episode}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create zip file containing all recordings
    zip_path = os.path.join(os.getcwd(), 'evaluation_recordings.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(video_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, video_dir)
                zipf.write(file_path, arcname)
    
    print("\nGenerated files:")
    print(f"Zip file: {zip_path}")
    print(f"Recordings directory: {video_dir}")
    
    return scores

def plot_scores(scores):
    """
    Plot the evaluation scores.
    
    Args:
        scores (list): List of evaluation scores to plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(scores, 'ro-')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Evaluation Scores')
    plt.show()

def main(config_path):
    """
    Main function to run the evaluation process.
    
    Args:
        config_path (str): Path to the configuration file
    """
    # Load configurations
    config = load_parameters(config_path)
    
    # Initialize environment with rgb_array render mode
    eval_env = gym.make(
        config['env']['name'],
        render_mode='rgb_array',
        **config['env']['Options']
    )
    
    # Create and load agent
    agent = Agent(config['agent'])
    agent.load_checkpoint(config['training']['checkpoint_path'])
    
    # Run evaluation
    eval_scores = eval_agent(agent, eval_env)
    
    # Plot results
    plot_scores(eval_scores)
    
    # Cleanup
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LunarLander DQN Evaluation")
    parser.add_argument(
        '--config',
        type=str,
        default='dqnconfig.yaml',
        help='Path to the configuration file (YAML format)'
    )
    args = parser.parse_args()
    main(args.config)