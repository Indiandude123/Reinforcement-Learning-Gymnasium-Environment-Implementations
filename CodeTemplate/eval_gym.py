import torch
import random
from os import system, name
from time import sleep
import moviepy.editor as mpy
import matplotlib.pyplot as plt

"""Display and evaluate agent's performance"""
q_table=torch.load("Sample_Q_table_FrozenNonSlippery.pt")
total_reward,total_steps=0,0
display_episodes=5
for i_episode in range(display_episodes):
    state, info = env.reset()
    rewards,steps= 0,0
    done = False
    frames = []
    frame = env.render()
    frames.append(frame)  # Collect frames for visualization.
        
    while not done:
        action = torch.argmax(q_table[state]).item()  # Use PyTorch to select the best action.
        state, reward, done, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)  # Collect frames for visualization.
        steps+=1
        rewards+=reward
        if(len(Traj)>500):
            break
    frame_rate = 1  # You can adjust this
    clip = mpy.ImageSequenceClip(frames, fps=frame_rate)
    clip.write_videofile(f"FrozenLake_NonSlippery_{i_episode}.mp4", codec="libx264")
    total_reward+=rewards
    total_steps+=steps
print(f"Results after {display_episodes} episodes:")
print(f"Average timesteps per episode: {total_steps / display_episodes}")
print(f"Average reward per episode: {total_reward / display_episodes}")
