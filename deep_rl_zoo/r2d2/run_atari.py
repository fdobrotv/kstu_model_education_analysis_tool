# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning"
https://openreview.net/pdf?id=r1lyTjAqYX.
"""

from absl import app
from absl import logging
import os
from flags import FLAGS

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing
import numpy as np
import torch
import copy

# pylint: disable=import-error
from deep_rl_zoo.networks.value import R2d2DqnConvNet, RnnDqnNetworkInputs
from deep_rl_zoo.r2d2 import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib

def main(argv):
    """Trains R2D2 agent on Atari."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs R2D2 agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Create evaluation environment, like R2D2, we disable terminate-on-life-loss and clip reward.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=True,
            sticky_action=False,
            clip_reward=False,
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.n

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Test environment and state shape.
    observation = eval_env.reset()
    # print(observation)
    #TODO: Check, why info is None here
    (obs, info) = observation
    
    # print("Before isinstance(obs, np.ndarray)")
    logging.info('Before isinstance(obs, np.ndarray)')
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)

    # Create network for learner to optimize, actor will use the same network with share memory.
    network = R2d2DqnConvNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, eps=FLAGS.adam_eps)

    # Test network output.
    x = RnnDqnNetworkInputs(
        s_t=torch.from_numpy(obs[None, None, ...]).float(),
        a_tm1=torch.zeros(1, 1).long(),
        r_t=torch.zeros(1, 1).float(),
        hidden_s=network.get_initial_hidden_state(1),
    )
    network_output = network(x)
    assert network_output.q_values.shape == (1, 1, action_dim)
    assert len(network_output.hidden_s) == 2

    # Create prioritized transition replay, no importance_sampling_exponent decay
    importance_sampling_exponent = FLAGS.importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent

    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                s_t=replay_lib.compress_array(transition.s_t),
            )

        def decoder(transition):
            return transition._replace(
                s_t=replay_lib.uncompress_array(transition.s_t),
            )

    else:
        encoder = None
        decoder = None

    logging.info('Before replay_lib.PrioritizedReplay(')
    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=agent.TransitionStructure,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
        time_major=True,
        encoder=encoder,
        decoder=decoder,
    )

    logging.info(f'FLAGS.num_actors: {FLAGS.num_actors}')
    # Create queue to shared transitions between actors and learner
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors * 2)

    logging.info('multiprocessing.Manager()')
    # Create shared objects so all actor processes can access them
    manager = multiprocessing.Manager()

    logging.info('Before manager.dict()')
    # Store copy of latest parameters of the neural network in a shared dictionary, so actors can later access it
    shared_params = manager.dict({'network': None})

    logging.info('Bedore agent.Learner(')
    # Create R2D2 learner instance
    learner_agent = agent.Learner(
        network=network,
        optimizer=optimizer,
        replay=replay,
        min_replay_size=FLAGS.min_replay_size,
        target_net_update_interval=FLAGS.target_net_update_interval,
        discount=FLAGS.discount,
        burn_in=FLAGS.burn_in,
        priority_eta=FLAGS.priority_eta,
        rescale_epsilon=FLAGS.rescale_epsilon,
        batch_size=FLAGS.batch_size,
        n_step=FLAGS.n_step,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
        shared_params=shared_params,
    )

    # Create actor environments, actor instances.
    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]

    actor_devices = ['cpu'] * FLAGS.num_actors
    # Evenly distribute the actors to all available GPUs
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

    print("Before agent.Actor")
    logging.info('Before agent.Actor')
    # Rank 0 is the most explorative actor, while rank N-1 is the most exploitative actor.
    # Each actor has it's own network with different weights.
    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            network=copy.deepcopy(network),
            random_state=np.random.RandomState(FLAGS.seed + int(i)),  # pylint: disable=no-member
            num_actors=FLAGS.num_actors,
            action_dim=action_dim,
            unroll_length=FLAGS.unroll_length,
            burn_in=FLAGS.burn_in,
            actor_update_interval=FLAGS.actor_update_interval,
            device=actor_devices[i],
            shared_params=shared_params,
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.R2d2EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='R2D2', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('network', network))

    # Run parallel training N iterations.
    # main_loop.run_parallel_training_iterations(
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_envs=actor_envs,
        data_queue=data_queue,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,
    )


# if __name__ == '__main__':
    # Set multiprocessing start mode
    # multiprocessing.set_start_method('spawn')
    # app.run(main)
