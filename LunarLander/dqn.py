import os
import numpy as np
import torch
from collections import deque
import wandb

def dqn(agent, env, TrainingConfigs):
    """Deep Q-Learning training loop"""
    rewards = []
    best_avg_score = -np.inf
    scores_window = deque(maxlen=TrainingConfigs['av_window'])
    
    def eps_fn(episode_i, eps_start=TrainingConfigs['eps_start'], 
               eps_end=TrainingConfigs['eps_end'], 
               eps_decay=TrainingConfigs['eps_decay']):
        return max(eps_end, eps_start * (eps_decay ** (episode_i)))
    
    for i_episode in range(1, TrainingConfigs['n_episodes'] + 1):
        state = env.reset()[0]
        score = 0
        eps = eps_fn(i_episode)
        
        for t in range(TrainingConfigs['max_traj_len']):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.add(state, action, reward, next_state, done)
            
            if len(agent.memory) > TrainingConfigs['batch_size']:
                experiences = agent.memory.sample(TrainingConfigs['batch_size'], agent.device)
                agent.batch_learn(experiences, TrainingConfigs['gamma'])
                
            state = next_state
            score += reward
            
            if done:
                break
                
        scores_window.append(score)
        rewards.append(score)
        
        if i_episode % TrainingConfigs['av_window'] == 0:
            avg_score = np.mean(scores_window)
            if TrainingConfigs['wandblogging']:
                wandb.log({
                    "Reward": avg_score,
                    "Episode": i_episode,
                    "Epsilon": eps
                })
                
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save_checkpoint(TrainingConfigs['checkpoint_path'])
                print('\r━━━> Episode {}\tAverage Score: {:.2f} | Saved'.format(
                    i_episode, avg_score))
            else:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    i_episode, avg_score))
                
            if avg_score >= 200.0:
                print('\n━━━━━━━━━━━━━━━ SOLVED ━━━━━━━━━━━━━━━')
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode - TrainingConfigs['av_window'], avg_score))
                return rewards
                
    return rewards