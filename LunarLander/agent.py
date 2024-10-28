import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Three fully connected layers
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.action_size = action_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size, device):
        experiences = random.sample(self.memory, k=batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self, configs):
        self.configs = configs
        self.state_size = configs['state_size']
        self.action_size = configs['action_size']
        self.device = configs['device']
        self.tau = configs['tau']
        
        # Q-Networks
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, configs['seed']).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, configs['seed']).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=configs['lr'])
        
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, configs['buffer_size'], configs['seed'])
        
        self.t_step = 0
        
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def batch_learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss and optimize
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.target_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
    def target_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save_checkpoint(self, path):
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return self.qnetwork_local