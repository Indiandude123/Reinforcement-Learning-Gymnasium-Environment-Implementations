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


def main(config_path):
    # Load all configurations from YAML file
    AllConfigs = load_parameters(config_path)
    AgentConfigs = AllConfigs['agent']
    TrainingConfigs = AllConfigs['training']
    EnvConfigs = AllConfigs['env']

    # Initialize wandb for visualization
    wandb.init(project="AIL722_A2",
               mode=TrainingConfigs['wandb_mode'],
               name=TrainingConfigs['wandb_run_name'])

    # Initialize the environment
    env = gym.make(EnvConfigs['name'], **EnvConfigs['Options'])

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    # Create agent instance
    agent = Agent(AgentConfigs)

    # Train the agent using DQN
    scores = dqn(agent, env, TrainingConfigs)

    # Plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    # Argument parser to get the config file path
    parser = argparse.ArgumentParser(description="LunarLander DQN Configuration")
    parser.add_argument('--config', type=str, default='dqnconfig.yaml',
                        help='Path to the configuration file (YAML format)')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the provided config file path
    main(args.config)
