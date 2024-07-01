#!/usr/bin/python3

import gymnasium as gym
from gymnasium.utils.play import play
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# from gymnasium import envs
# all_envs = envs.registry
# print(all_envs)
# for x in all_envs:
#     print (x)

number_of_environments = 4
seed_val = 0

tensorflow_path = "/tf_logs/"
# monitor_path = f"{tensorflow_path}{timestr}_{env_name}_monitor/"
frame_stack = 4


# def create_vec_env_with_difficulty(difficulty):
#   vec_env = make_atari_env(env_name, n_envs=number_of_environments, seed=seed_val, env_kwargs=dict(difficulty=difficulty))
#   vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
#   return vec_env

def create_env(difficulty):
  vec_env = make_atari_env(env_name, n_envs=number_of_environments, seed=seed_val, env_kwargs=dict(difficulty=difficulty))
  vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
  # env = gym.make(, render_mode='rgb_array', difficulty=difficulty)
  return vec_env

def load_model_with_difficulty(path, difficulty):
  vec_env = create_env(difficulty)
  model = PPO.load(path, vec_env)
  return model
      
def model_to_images(model):
  images = []
  obs = model.env.reset()
  img = model.env.render(mode="human")
  for i in range(1500):
      images.append(img)
      action, _ = model.predict(obs)
      obs, _, _ ,_ = model.env.step(action)
      img = model.env.render(mode="human")
  return images

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 144)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    plt.close(anim._fig)
    display(HTML(anim.to_jshtml()))

def playByModel(model_path, difficulty):
  model = load_model_with_difficulty(model_path, difficulty)
  images = model_to_images(model)
  display_frames_as_gif(images)
  
# env_id = "ALE/Pong"
# env_name = env_id + "-v5"

# env_id = "Pong"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 3
# playByModel("models/PongNoFrameskip-v4_20240220-100148_28254832_steps.zip", max_difficulty)

# env_id = "Breakout"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 1
# playByModel("models/BreakoutNoFrameskip-v4_20240220-122359_best/best_model.zip", max_difficulty)
# playByModel("models/BreakoutBreakoutNoFrameskip-v4_20240221-113055_best/best_model.zip", max_difficulty)
# playByModel("models/Breakout/BreakoutNoFrameskip-v4_20240221-113055_7700000_steps.zip", max_difficulty)

# env_id = "MsPacman"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# playByModel("models/MsPacmanNoFrameskip-v4_20240221-104542_best/best_model.zip", max_difficulty)

# env_id = "Qbert"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 1
# playByModel("models/QbertQbertNoFrameskip-v4_20240222-110348_best/best_model.zip", max_difficulty)

env_id = "Robotank"
env_name = env_id + "NoFrameskip-v4"
max_difficulty = 0
playByModel(f"models/{env_id}{env_id}NoFrameskip-v4_20240225-100948_best/best_model.zip", max_difficulty)