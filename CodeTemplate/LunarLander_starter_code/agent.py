import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    """Define Neural Network Architecture"""

    def __init__(self, state_size, action_size,seed):
        """
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        ###Your Code Here###
        


    def forward(self, state):
        """Build a network that maps state -> action values."""
        ###Your Code Here###
        
        
        pass


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        ####Your Code Here ####
        #self.memory =

    
    def add(self, state, action, reward, next_state, done):
       """Add a new experience to memory."""
        ####Your Code Here ####
        
    def sample(self, batch_size,device):
        """Randomly sample a batch of experiences from memory.
        Return as tensors with device as required
        """
        ####Your Code Here ####
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, configs):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.configs=configs
        self.state_size = configs['state_size']
        self.action_size = configs['action_size']
        random.seed(configs['seed'])
        self.device=configs['device']
        self.tau =configs['tau']
        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, configs['seed']).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, configs['seed']).to(self.device)
        ####Your Code Here ####

        
        #Setup Optimizer
        #self.optimizer = 
        
        #Setup Replay memory
        
        #self.memory = ReplayBuffer(self.action_size, configs['buffer_size'], configs['seed'])
        
        self.t_step = 0
        
    def save_checkpoint(self,Path):
        """Save Q_network parameters"""
        #print(f"Saved Checkpoint at {Path} !")
        ####Your Code Here ####

    def load_checkpoint(self,Path):
        """Load Q_network parameters"""
        #print(f"Loaded Checkpoint from {Path} !")
        ####Your Code Here ####
        
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        
        Use eps=0 for final deterministic policy
        """
        
        # Epsilon-greedy action selection

        ####Your Code Here ####
        
        return action
    def batch_learn(self, experiences, gamma):
        """
        This method updates the value parameters of the neural network using one batch of experience tuples.

        Arguments:
        - experiences (Tuple[torch.Variable]): A tuple containing five elements:
            * s: The states (tensor)
            * a: The actions (tensor)
            * r: The rewards (tensor)
            * s': The next states (tensor)
            * done: Boolean tensors indicating if the episode has terminated (tensor)

        - gamma (float): The discount factor used in the Q-learning update rule (also called Bellman equation).

        Returns:
        None
        """

        ####Your Code Here ####

        

    def target_update(self, local_model, target_model, tau):
        """Soft update time delayed target model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied 
            target_model (PyTorch model): weights will be copied to
        """

        ####Your Code Here ####




        