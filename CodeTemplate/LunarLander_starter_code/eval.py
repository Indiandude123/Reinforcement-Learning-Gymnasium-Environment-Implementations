import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent import *
from dqn import *
import os
from collections import deque

import gymnasium as gym
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
import argparse


def load_parameters(file_path):
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


def eval(agent,env,TrainingConfigs):
    """ Evaluate the agent
    """
    scores = []  # list containing scores from each episode
    
    for i_episode in range(1, 50):
        state = env.reset()[0]
        score = 0
        frames=[]
        for t in range(1500):
            action = agent.act(state, eps=0.0) #Greedy deterministic policy
            observation, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            state = observation
            score += reward
            if terminated:
                break
        scores.append(score)  # save most recent score
        
        if i_episode % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            frame_rate = 30  # You can adjust this
            clip = mpy.ImageSequenceClip(frames, fps=frame_rate)
            clip.write_videofile(f"Lunar_lander_trajectory{i_episode}.mp4", codec="libx264")
            
    return scores

def main(config_path):
    # Load all configurations from YAML file
    AllConfigs = load_parameters(config_path)
    AgentConfigs = AllConfigs['agent']
    TrainingConfigs = AllConfigs['training']
    EnvConfigs = AllConfigs['env']

    
    # Initialize the environment
    env = gym.make(EnvConfigs['name'], **EnvConfigs['Options'])

    # Create agent instance
    agent = Agent(AgentConfigs)
    agent.load_checkpoint(TrainingConfigs['checkpoint_path']).to()
    
    eval(agent,env,TrainingConfigs)
    

if __name__ == "__main__":
    # Argument parser to get the config file path
    parser = argparse.ArgumentParser(description="LunarLander DQN Configuration")
    parser.add_argument('--config', type=str, default='dqnconfig.yaml',
                        help='Path to the configuration file (YAML format)')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the provided config file path
    main(args.config)
