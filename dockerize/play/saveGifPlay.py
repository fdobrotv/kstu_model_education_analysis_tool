import imageio
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

number_of_environments = 4
seed_val = 0

tensorflow_path = "/tf_logs/"
frame_stack = 4

def create_env(env_name, difficulty):
  vec_env = make_atari_env(env_name, n_envs=number_of_environments, seed=seed_val, env_kwargs=dict(difficulty=difficulty))
  vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
  return vec_env

def create_env_with_video(env_name, algo_name, difficulty, video_save_path, video_length):
  vec_env = make_atari_env(env_name, n_envs=number_of_environments, seed=seed_val, env_kwargs=dict(difficulty=difficulty))
  vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
  vec_env = VecVideoRecorder(vec_env, video_save_path,
                        record_video_trigger=lambda x: x == 0, 
                        video_length=video_length,
                        name_prefix=f"best_agent_{env_name}_{algo_name}")
  return vec_env

def load_model_with_video_and_difficulty(algo, algo_name, env_name, path, difficulty, video_save_path, video_length):
  vec_env = create_env_with_video(env_name, algo_name, difficulty, video_save_path, video_length)
  model = algo.load(path, vec_env)
  return model

def load_model_with_difficulty(algo, env_name, path, difficulty):
  vec_env = create_env(env_name, difficulty)
  model = algo.load(path, vec_env)
  return model


def saveGifPlay(algo, algo_name, env_name, video_length, model_path, difficulty):
    model = load_model_with_difficulty(algo, env_name, f"{model_path}/best_model", difficulty)
    images = []
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")
    for i in range(video_length):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render(mode="rgb_array")

    imageio.mimsave(f"{model_path}{env_name}_{algo_name}.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=24)
   
def saveGifPlayV2(algo, algo_name, env_name, video_length, model_path, save_path, difficulty):
    model = load_model_with_difficulty(algo, env_name, model_path, difficulty)
    images = []
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")
    for i in range(video_length):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render(mode="rgb_array")

    imageio.mimsave(f"{save_path}{env_name}_{algo_name}.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=24)
     
def saveVideoPlay(algo, algo_name, env_name, video_length, video_save_path, difficulty):
  # vec_env = DummyVecEnv([lambda: gym.make(env_name, render_mode="rgb_array", difficulty=difficulty)])
  model = load_model_with_video_and_difficulty(algo, algo_name, env_name, f"{video_save_path}/best_model", difficulty, video_save_path, video_length)
  vec_env = model.env
  # obs = vec_env.reset()
  # Record the video starting at the first step
  # vec_env = VecVideoRecorder(vec_env, model_path,
  #                       record_video_trigger=lambda x: x == 0, 
  #                       video_length=video_length,
  #                       name_prefix=f"best_agent_{env_name}_{algo_name}")

  vec_env.reset()
  for _ in range(video_length + 1):
    action = [vec_env.action_space.sample()]
    obs, _, _, _ = vec_env.step(action)
  # Save the video
  vec_env.close()

# env_id = "Robotank"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 3000
# saveGifPlay(env_name, length, f"models/{env_id}{env_id}NoFrameskip-v4_20240225-100948_best/", max_difficulty)
# saveVideoPlay(env_name, length, f"models/{env_id}{env_id}NoFrameskip-v4_20240225-100948_best/", max_difficulty)


# env_id = "Gravitar"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 3000
# best_model_path = f"models/{env_id}{env_id}NoFrameskip-v4_20240225-102117_best/"
# saveGifPlay(env_name, length, best_model_path, max_difficulty)
# saveVideoPlay(env_name, length, best_model_path, max_difficulty)


# env_id = "Breakout"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 4000
# algo = A2C
# algo_name = "A2C"
# best_model_path = f"models/20240305-125531_{env_name}_A2CCNN4FrameStackHuggingFaceParametersV7_20240305-125531_best/"
# saveVideoPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)
# saveGifPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)

# env_id = "BeamRider"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 4000
# algo = DQN
# algo_name = "DQN"
# best_model_path = f"models/20240309-144503_{env_name}_{algo_name}_CnnPolicy_4-FrameStack_1kkBF_Optuna_V10_best/"
# saveGifPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)
# saveVideoPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)

# env_id = "Pong"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 4000
# algo = PPO
# algo_name = "PPO"
# best_model_path = f"models/20240321-164641_{env_name}_{algo_name}_CnnPolicy_4-FrameStack_Seed2_V12_best/"
# saveGifPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)
# saveVideoPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)


# env_id = "Breakout"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 4000
# algo = A2C
# algo_name = "A2C"
# best_model_path = f"models/20240320-143834_{env_name}_{algo_name}_CnnPolicy_4-FrameStack_Seed2_V12_best/"
# saveGifPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)
# saveVideoPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)

# env_id = "Seaquest"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 4000
# algo = PPO
# algo_name = "PPO"
# best_model_path = f"models/20240316-113020_{env_name}_{algo_name}_CnnPolicy_4-FrameStack_1kkBF_V11_best/"
# saveGifPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)
# saveVideoPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)

# env_id = "Frostbite"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 4000
# algo = PPO
# algo_name = "PPO"
# last_model = "FrostbiteNoFrameskip-v4_20240312-142240_20000000_steps.zip"
# model_path = f"models/20240312-142240_{env_name}_{algo_name}_CnnPolicy_4-FrameStack_1kkBF_V11"
# best_model_path = f"{model_path}_best/"
# full_model_path = f"{model_path}/{last_model}"
# save_path = "gifs/"
# saveGifPlayV2(algo, algo_name, env_name, length, full_model_path, save_path, max_difficulty)
# # saveVideoPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)


# env_id = "Frostbite"
# env_name = env_id + "NoFrameskip-v4"
# max_difficulty = 0
# length = 4000
# algo = PPO
# algo_name = "PPO"
# best_model_path = f"models/20240312-142240_{env_name}_{algo_name}_CnnPolicy_4-FrameStack_1kkBF_V11_best/"
# saveGifPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)
# saveVideoPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)


env_id = "BeamRider"
env_name = env_id + "NoFrameskip-v4"
max_difficulty = 0
length = 4000
algo = A2C
algo_name = "A2C"
best_model_path = f"models/20240306-125608_{env_name}_{algo_name}_CnnPolicy_4-FrameStack_V9_best/"
saveGifPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)
saveVideoPlay(algo, algo_name, env_name, length, best_model_path, max_difficulty)