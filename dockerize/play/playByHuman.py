#!/usr/bin/python3

import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium import envs

# all_envs = envs.registry
# print(all_envs)
# for x in all_envs:
#     print (x)

# env_id = "Pong"
env_id = "MsPacman"
env_mode = "NoFrameskip"
env_name = env_id + f"{env_mode}-v4"
env_name_ale = f"ALE/{env_name}"

# metadata = {'render_fps': 60}
env = gym.make(env_name, render_mode='rgb_array', mode=0, difficulty=0)
print(env.metadata)
play(env, zoom=3, fps=24)