import gymnasium as gym
import torch
import random
import numpy as np
from os import system, name
from time import sleep
from tqdm import tqdm
import wandb

def Algorithm(env, gamma=0.95,alpha=0.05,MaxEpisodes=20000):
    #Intialize the state-action value function
    NStates=env.observation_space.n
    NActions=env.action_space.n
    #wandb.init(project="AIL722_A2", mode='online',name='Env_Algo_Run_k')
    
    Q=torch.zeros([env.observation_space.n, env.action_space.n])
    ########################Your Code Here ####################
    
    #for episode in range(MaxEpisodes):
    
    
        #if (episode+1)%K==0:
         #  print("Episode: ",episode,"EPS:",np.round(EPS,2),"MeanRewards: ",MeanRewards)
            #wandb.log({"MeanRewards":MeanRewards,"EPS":EPS,"Episode":episode})
            
    #wandb.finish()

   
    ############################################################
    return Q

#env = gym.make('Taxi-v3', render_mode='rgb_array') # Setup the Gym Environment #env=gym.make('FrozenLake-v1', desc=None, map_name="4x4",is_slippery=False,render_mode='rgb_array')
#q_table=QLearning_TD(env,MaxEpisodes=20000)
#torch.save(q_table,"Env_Algo_RunName.pt")