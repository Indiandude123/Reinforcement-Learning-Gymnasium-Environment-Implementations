from env import TreasureHunt_v2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torch 
import torch.nn as nn
import random 
import os 
import wandb
wandb.init(mode="offline", project="treasurehuntv2_qdn")

class CacheData:

    def __init__(self, cache_capacity = 100000):
        self.cache_capacity = cache_capacity
        self.data = []
        self.qprob = []
        
    
    def add_item(self, item):

        #ADD YOUR LOGIC HERE

    def sample_batch(self, batch_size):
        
        #SAMPLE THE TUPLE OF BATCH SIZE
        #SINGLE TUPLE IS STATE, ACTION, NEXT_STATE, REWARD, WEIGHT


class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 64, (3,3), 1, 1)
        self.conv2 = nn.Conv2d(64, 64, (3,3), padding = 1, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, (3,3), padding = 1, stride = 2)
        
        self.fc1 = nn.Linear(64*9, 64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()

    def forward(self, x, actions = None):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.flatten(1,-1)
        x = self.fc2(self.relu(self.fc1(x)))
        return x 
        
        

class DeepQAgent:

    def __init__(self, env, demo_env, cache_size = 100000):

        self.env = env 
        self.dataset = CacheData(cache_size)
        self.qnetwork = QNetwork()
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr = 0.00005)
        
        #TO VALIDATE
        self.demo_env = demo_env
        

    def get_policy(self, states):
        states = torch.tensor(states).float()
        qsa = self.qnetwork.get_all(states)
        qsa = qsa.argmax(dim = -1)
        qsa = torch.nn.functional.one_hot(qsa)
        policy = qsa.numpy()
        return policy

    def visualize_policy(self, itr, path = './treasurehunt_v2_dqn/'):
        os.makedirs(path, exist_ok = True)
        for i, e in enumerate(self.demo_env[:3]):
            states = e.get_all_states()
            policy = self.get_policy(states)
            path_ = os.path.join(path, f'visualize_policy_{i}_{itr}.png')
            e.visualize_policy(policy, path_)

            path_ = os.path.join(path, f'visualize_policy_{i}_{itr}.gif')
            e.visualize_policy_execution(policy, path_)

    def validate(self):

        #YOUR CODE TO VALIDATE THE POLICY
        pass
        

    def choose_action(self, state):
        
        #CHOOSE THE EPSILON GREEDY ACCURACY

    def learn_policy(self, itrs = 50000):
        
        losses = []
        for i in tqdm(range(1, itrs)):

            #reset the state
            state = self.env.reset()

            #add episode to the buffer
            for j in range(100):

                action = self.choose_action(state)
                next_state_, reward = self.env.step(action)
                self.dataset.add_item((state, action, next_state_, reward))
                state = next_state_

            
            state, action, next_state, reward, weights = self.dataset.sample_batch(256)
            
            #YOUR LOGIC TO BACKPROPOGATE THE LOSS THROUGH Q NETWORK

            
            #EPSILON DECAY STEP

            if(i % 1000 == 0):
                self.visualize_policy(i)
                demo_values = self.qnetwork.get_all(self.demo_state).max().item()
                rewards = self.validate()
                wandb.log({'rewards': rewards, "steps": i})
                print(f"loss: {sum(losses)/len(losses)}  reward: {rewards} eps: {self.eps} buffer reward: {self.dataset.average_reward()}")
                losses = []



demo_envs = []
for i in range(100):
    demo_envs.append(TreasureHunt_v2())    
env = TreasureHunt_v2()
qagent = DeepQAgent(env, demo_envs)
qagent.learn_policy()
