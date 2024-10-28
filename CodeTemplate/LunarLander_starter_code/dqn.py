import os
import numpy as np
import torch
from collections import deque
import wandb

def dqn(agent,env,TrainingConfigs):
    """Deep Q-Learning.        
    """
    rewards = []  # list containing scores from each episode

    ####Your Code Here ####
    
    def eps_fn(episode_i,eps_start=TrainingConfigs['eps_start'],eps_end=TrainingConfigs['eps_end'],eps_decay=TrainingConfigs['eps_decay']):
        ####Your Code Here ####
        return eps
    
        
        
    for i_episode in range(1, TrainingConfigs['n_episodes'] + 1):
        eps=eps_fn(i_episode)    
        rewards.append(reward)  # save most recent score
        
        
        ####Your Code Here ####
        
        if i_episode % TrainingConfigs['av_window'] == 0:
            ####Your Code Here ###



            
            if TrainingConfigs['wandblogging']:
                wandb.log({"Reward" : avg_rewards ,"Episode":i_episode, "Epsilon":eps})
            if avg_score > best_avg_score:  
                best_avg_score = avg_score
                agent.save_checkpoint(TrainingConfigs['checkpoint_path'])
                print('\r━━━> Episode {}\tAverage Score: {:.2f} | Saved'.format(i_episode, avg_rewards))
            else:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_rewards))

        if np.mean(rewards[-TrainingConfigs['av_window']:]) >= 200.0:
            print('\n━━━━━━━━━━━━━━━ SOLVED ━━━━━━━━━━━━━━━')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - TrainingConfigs['av_window'],
                                                                                         avg_score))
    return rewards