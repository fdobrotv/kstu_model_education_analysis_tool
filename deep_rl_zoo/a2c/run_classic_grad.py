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
Notes:
    * Actors sample batch of transitions to calculate loss, but not optimization step.
    * Actors collects local gradients, and send to master through multiprocessing.Queue.
    * Learner will aggregates batch of gradients then do the optimization step.
    * Learner update policy network parameters for actors.

Note only supports training on single machine.

From the paper "Asynchronous Methods for Deep Reinforcement Learning"
https://arxiv.org/abs/1602.01783

Synchronous, Deterministic variant of A3C
https://openai.com/blog/baselines-acktr-a2c/.

"""
from absl import app
from absl import flags
from absl import logging
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing
import numpy as np
import torch
import copy

# pylint: disable=import-error
from deep_rl_zoo.networks.policy import ActorCriticMlpNet
from deep_rl_zoo.a2c import agent_grad as agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1.',
)
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_bool('compress_gradient', True, 'Actor process to compress the local gradients before put on queue, default on.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('entropy_coef', 0.0025, 'Coefficient for the entropy loss.')
flags.DEFINE_float('value_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_integer('batch_size', 32, 'Accumulate batch size transitions before do learning.')
flags.DEFINE_integer('learner_batch_size', 16, 'Accumulate batch size of gradients before do back-propagation.')
flags.DEFINE_integer('num_iterations', 2, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(5e5), 'Number of training env steps to run per iteration, per actor.')
flags.DEFINE_integer('num_eval_steps', int(2e4), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/a2c_grad_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')


def main(argv):
    """Trains A2C agent on classic control tasks."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs A2C agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Create environment.
    def environment_builder():
        return gym_env.create_classic_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**10),
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.n

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Create policy network and optimizer
    policy_network = ActorCriticMlpNet(state_dim=state_dim, action_dim=action_dim)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    # Test network output.
    obs = eval_env.reset()
    s = torch.from_numpy(obs[None, ...]).float()
    network_output = policy_network(s)
    assert network_output.pi_logits.shape == (1, action_dim)
    assert network_output.value.shape == (1, 1)

    # Create gradient replay, which is just a simple list with some optimization to speed up gradient aggregation
    gradient_replay = replay_lib.GradientReplay(FLAGS.num_actors, policy_network, FLAGS.compress_gradient)

    # Create queue to shared transitions between actors and learner
    gradient_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)
    lock = multiprocessing.Lock()

    # Create shared objects so all actor processes can access them
    manager = multiprocessing.Manager()

    # Store copy of latest parameters of the neural network in a shared dictionary, so actors can later access it
    shared_params = manager.dict({'policy_network': None})

    # Create A2C learner agent instance
    learner_agent = agent.Learner(
        lock=lock,
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        gradient_replay=gradient_replay,
        batch_size=FLAGS.learner_batch_size,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
        shared_params=shared_params,
    )

    # Create actor environments, runtime devices, and actor instances.
    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]
    actor_devices = ['cpu'] * FLAGS.num_actors
    # Evenly distribute the actors to all available GPUs
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

    actors = [
        agent.Actor(
            rank=i,
            lock=lock,
            gradient_queue=gradient_queue,
            policy_network=copy.deepcopy(policy_network),
            transition_accumulator=replay_lib.TransitionAccumulator(),
            discount=FLAGS.discount,
            batch_size=FLAGS.batch_size,
            compress_gradient=FLAGS.compress_gradient,
            entropy_coef=FLAGS.entropy_coef,
            value_coef=FLAGS.value_coef,
            clip_grad=FLAGS.clip_grad,
            max_grad_norm=FLAGS.max_grad_norm,
            device=actor_devices[i],
            shared_params=shared_params,
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='A2C-GRAD',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name='A2C-GRAD', save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('policy_network', policy_network))

    # Run parallel training N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_envs=actor_envs,
        data_queue=gradient_queue,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
