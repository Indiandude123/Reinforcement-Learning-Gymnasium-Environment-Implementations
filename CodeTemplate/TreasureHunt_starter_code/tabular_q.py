from env import TreasureHunt_v1
import numpy as np
import random 
import os 


class TabularQAgent:

    def __init__(self, env):
        self.episode_len = 100
        self.env = env

        ## YOU ARE FREE TO ADD YOUR OWN CODE HERE
       
        
    def choose_action(self, state):
        '''
        choose a epsilon greedy action for state 
        '''
        pass

    def get_policy(self):
        policy = np.random.randint(4, size = 400)
        for i in tqdm(range(120000)):
            s = self.env.reset()
            for t in tqdm(range(self.episode_len), leave = False):
                '''
                ADD YOUR LOGIC HERE
                '''
        #the value function
        return policy



env = TreasureHunt_v1()
agent = TabularQAgent(env)
policy, value = agent.get_policy()
env.visualize_policy(policy, "qlearning.png")
env.visualize_policy_execution(policy, "qlearning.gif")
