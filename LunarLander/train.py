import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent import Agent
from dqn import dqn
import os
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import wandb
import yaml
import argparse

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def main(config_path):
    # Load configurations
    AllConfigs = load_parameters(config_path)
    AgentConfigs = AllConfigs['agent']
    TrainingConfigs = AllConfigs['training']
    EnvConfigs = AllConfigs['env']
    
    # Initialize wandb
    wandb.init(
        project="AIL722_A2",
        mode=TrainingConfigs['wandb_mode'],
        name=TrainingConfigs['wandb_run_name']
    )
    
    # Initialize environment
    env = gym.make(EnvConfigs['name'], **EnvConfigs['Options'])
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    
    # Create and train agent
    agent = Agent(AgentConfigs)
    scores = dqn(agent, env, TrainingConfigs)
    
    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LunarLander DQN Configuration")
    parser.add_argument('--config', type=str, default='dqnconfig.yaml',
                        help='Path to the configuration file (YAML format)')
    args = parser.parse_args()
    main(args.config)